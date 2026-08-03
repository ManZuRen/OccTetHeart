from __future__ import annotations

import json
from argparse import ArgumentParser
from pathlib import Path

import meshio
import nibabel as nib
import numpy as np
import torch


def _load_label_volume(nifti_path: str) -> dict:
    image = nib.load(nifti_path)
    data_xyz = np.asarray(image.get_fdata(dtype=np.float32), dtype=np.float32)
    if data_xyz.ndim == 4:
        data_xyz = np.squeeze(data_xyz)
    if data_xyz.ndim != 3:
        raise ValueError(f"Expected 3D NIfTI volume, got shape {data_xyz.shape}.")

    affine = np.asarray(image.affine, dtype=np.float32)
    basis = affine[:3, :3]
    spacing = np.linalg.norm(basis, axis=0).astype(np.float32)
    safe_spacing = np.where(spacing > 0.0, spacing, 1.0).astype(np.float32)
    direction = (basis / safe_spacing[None, :]).astype(np.float32)

    return {
        "image": image,
        "array_zyx": np.transpose(data_xyz, (2, 1, 0)),
        "spacing": spacing,
        "direction": direction,
        "inv_direction": np.linalg.inv(direction).astype(np.float32),
        "size_xyz": np.asarray(data_xyz.shape, dtype=np.int32),
    }


def _read_tetra_mesh(mesh_path: str) -> tuple[np.ndarray, np.ndarray]:
    mesh = meshio.read(mesh_path)
    vertices = np.asarray(mesh.points[:, :3], dtype=np.float64)
    tetra_blocks = [np.asarray(c.data, dtype=np.int64) for c in mesh.cells if c.type == "tetra"]
    if not tetra_blocks:
        raise ValueError(f"No tetra cells found in {mesh_path}.")
    cells = np.concatenate(tetra_blocks, axis=0)
    return vertices, cells


def _extract_boundary_faces(cells: np.ndarray, vertices: np.ndarray) -> np.ndarray:
    face_patterns = np.asarray(
        [
            [0, 1, 2],
            [0, 3, 1],
            [0, 2, 3],
            [1, 3, 2],
        ],
        dtype=np.int64,
    )
    opposite_patterns = np.asarray([3, 2, 1, 0], dtype=np.int64)
    oriented_faces = []
    face_keys = []

    for cell in np.asarray(cells, dtype=np.int64):
        tet = np.asarray(vertices[cell], dtype=np.float64)
        for face_pattern, opposite_pattern in zip(face_patterns, opposite_patterns):
            face = np.asarray(cell[face_pattern], dtype=np.int64).copy()
            p0, p1, p2 = np.asarray(vertices[face], dtype=np.float64)
            opp = tet[int(opposite_pattern)]
            normal = np.cross(p1 - p0, p2 - p0)
            # Boundary normals must point away from the tetra interior.
            if float(np.dot(normal, opp - p0)) > 0.0:
                face[[1, 2]] = face[[2, 1]]
            oriented_faces.append(face)
            face_keys.append(tuple(sorted(int(v) for v in face)))

    counts = {}
    for key in face_keys:
        counts[key] = counts.get(key, 0) + 1

    boundary_faces = [face for face, key in zip(oriented_faces, face_keys) if counts[key] == 1]
    if not boundary_faces:
        return np.empty((0, 3), dtype=np.int64)
    return np.asarray(boundary_faces, dtype=np.int64)


def _centered_grid_origin_xyz(size_xyz: np.ndarray, spacing_xyz: np.ndarray) -> np.ndarray:
    return (-0.5 * (np.asarray(size_xyz, dtype=np.float64) - 1.0) * np.asarray(spacing_xyz, dtype=np.float64)).astype(
        np.float64
    )


def _make_grid_points_zyx(shape_zyx: tuple[int, int, int], origin_xyz: np.ndarray, spacing_xyz: np.ndarray) -> np.ndarray:
    z_count, y_count, x_count = [int(v) for v in shape_zyx]
    xs = origin_xyz[0] + np.arange(x_count, dtype=np.float64) * spacing_xyz[0]
    ys = origin_xyz[1] + np.arange(y_count, dtype=np.float64) * spacing_xyz[1]
    zs = origin_xyz[2] + np.arange(z_count, dtype=np.float64) * spacing_xyz[2]
    grid_z, grid_y, grid_x = np.meshgrid(zs, ys, xs, indexing="ij")
    return np.stack([grid_x, grid_y, grid_z], axis=-1).reshape(-1, 3)


def _solid_angle_sum(points: np.ndarray, triangles: np.ndarray, face_chunk_size: int) -> np.ndarray:
    winding = np.zeros((points.shape[0],), dtype=np.float64)
    eps = 1e-15

    for start in range(0, triangles.shape[0], int(face_chunk_size)):
        tri = triangles[start : start + int(face_chunk_size)]
        a = tri[None, :, 0, :] - points[:, None, :]
        b = tri[None, :, 1, :] - points[:, None, :]
        c = tri[None, :, 2, :] - points[:, None, :]

        la = np.linalg.norm(a, axis=-1)
        lb = np.linalg.norm(b, axis=-1)
        lc = np.linalg.norm(c, axis=-1)
        det = np.einsum("...i,...i->...", a, np.cross(b, c))
        denom = (
            la * lb * lc
            + np.einsum("...i,...i->...", a, b) * lc
            + np.einsum("...i,...i->...", b, c) * la
            + np.einsum("...i,...i->...", c, a) * lb
        )
        winding += np.sum(2.0 * np.arctan2(det, denom + eps), axis=1)

    return winding


def _solid_angle_sum_torch(
    points: torch.Tensor,
    triangles: torch.Tensor,
    face_chunk_size: int,
) -> torch.Tensor:
    winding = torch.zeros((points.shape[0],), dtype=points.dtype, device=points.device)
    eps = torch.finfo(points.dtype).eps

    for start in range(0, triangles.shape[0], int(face_chunk_size)):
        tri = triangles[start : start + int(face_chunk_size)]
        a = tri[None, :, 0, :] - points[:, None, :]
        b = tri[None, :, 1, :] - points[:, None, :]
        c = tri[None, :, 2, :] - points[:, None, :]

        la = torch.linalg.norm(a, dim=-1)
        lb = torch.linalg.norm(b, dim=-1)
        lc = torch.linalg.norm(c, dim=-1)
        det = torch.sum(a * torch.cross(b, c, dim=-1), dim=-1)
        denom = (
            la * lb * lc
            + torch.sum(a * b, dim=-1) * lc
            + torch.sum(b * c, dim=-1) * la
            + torch.sum(c * a, dim=-1) * lb
        )
        winding = winding + torch.sum(2.0 * torch.atan2(det, denom + eps), dim=1)

    return winding


def points_inside_closed_surface(
    points: np.ndarray,
    surface_vertices: np.ndarray,
    surface_faces: np.ndarray,
    point_chunk_size: int,
    face_chunk_size: int,
    winding_threshold: float,
) -> np.ndarray:
    triangles = np.asarray(surface_vertices[surface_faces], dtype=np.float64)
    inside = np.zeros((points.shape[0],), dtype=bool)

    for start in range(0, points.shape[0], int(point_chunk_size)):
        stop = min(start + int(point_chunk_size), points.shape[0])
        winding = _solid_angle_sum(points[start:stop], triangles, face_chunk_size=int(face_chunk_size))
        winding_number = np.abs(winding) / (4.0 * np.pi)
        inside[start:stop] = winding_number >= float(winding_threshold)
        print(f"[inside] points {start}:{stop} inside={int(inside[start:stop].sum())}")

    return inside


def points_inside_closed_surface_torch(
    points: np.ndarray,
    surface_vertices: np.ndarray,
    surface_faces: np.ndarray,
    point_chunk_size: int,
    face_chunk_size: int,
    winding_threshold: float,
    device: str,
) -> np.ndarray:
    triangles_np = np.asarray(surface_vertices[surface_faces], dtype=np.float32)
    triangles = torch.as_tensor(triangles_np, dtype=torch.float32, device=device)
    inside = np.zeros((points.shape[0],), dtype=bool)

    for start in range(0, points.shape[0], int(point_chunk_size)):
        stop = min(start + int(point_chunk_size), points.shape[0])
        points_chunk = torch.as_tensor(points[start:stop].astype(np.float32, copy=False), dtype=torch.float32, device=device)
        winding = _solid_angle_sum_torch(points_chunk, triangles, face_chunk_size=int(face_chunk_size))
        winding_number = torch.abs(winding) / (4.0 * torch.pi)
        inside_chunk = winding_number >= float(winding_threshold)
        inside[start:stop] = inside_chunk.detach().cpu().numpy()
        print(f"[inside:{device}] points {start}:{stop} inside={int(inside[start:stop].sum())}")

    return inside


def _dice(pred: np.ndarray, gt: np.ndarray) -> float:
    pred_f = pred.astype(np.float64)
    gt_f = gt.astype(np.float64)
    inter = float(np.sum(pred_f * gt_f))
    denom = float(np.sum(pred_f) + np.sum(gt_f))
    return (2.0 * inter + 1.0) / (denom + 1.0 + 1e-6)


def _save_nifti_like(array_zyx: np.ndarray, reference: nib.spatialimages.SpatialImage, output_path: Path):
    array_xyz = np.ascontiguousarray(np.transpose(array_zyx, (2, 1, 0)).astype(np.uint8))
    out = nib.Nifti1Image(array_xyz, np.asarray(reference.affine, dtype=np.float32), reference.header.copy())
    out.header.set_data_shape(array_xyz.shape)
    out.header.set_data_dtype(np.uint8)
    nib.save(out, str(output_path))


def parse_args():
    parser = ArgumentParser(description="Compute hard 3D Dice by exact geometric point-in-tetra-boundary test.")
    parser.add_argument("--mesh", required=True, help="Input tetrahedral .msh.")
    parser.add_argument("--label-nifti", required=True, help="Reference label NIfTI.")
    parser.add_argument("--label-value", type=int, default=2, help="Foreground label. Use -1 for any nonzero label.")
    parser.add_argument("--target-rescale", type=float, default=0.01, help="Scale applied to NIfTI spacing, matching OccTetHeart training.")
    parser.add_argument("--mesh-is-local", action="store_true", help="Mesh is already in OccTetHeart centered local-grid coordinates.")
    parser.add_argument("--winding-threshold", type=float, default=0.5)
    parser.add_argument("--backend", type=str, default="auto", choices=("auto", "torch_cuda", "numpy"))
    parser.add_argument("--point-chunk-size", type=int, default=8192)
    parser.add_argument("--face-chunk-size", type=int, default=1024)
    parser.add_argument("--output-dir", type=str, default=None, help="Optional directory for pred/gt masks and summary.")
    return parser.parse_args()


def main():
    args = parse_args()
    volume = _load_label_volume(args.label_nifti)
    vertices, cells = _read_tetra_mesh(args.mesh)
    if not args.mesh_is_local:
        vertices = vertices @ np.asarray(volume["inv_direction"], dtype=np.float64).T

    boundary_faces = _extract_boundary_faces(cells, vertices)
    spacing_xyz = np.asarray(volume["spacing"], dtype=np.float64) * float(args.target_rescale)
    origin_xyz = _centered_grid_origin_xyz(volume["size_xyz"], spacing_xyz)
    gt_zyx = np.asarray(volume["array_zyx"], dtype=np.float32)
    gt_mask = gt_zyx > 0.0 if int(args.label_value) < 0 else gt_zyx == float(args.label_value)

    print(
        "[mesh] "
        f"vertices={vertices.shape[0]} tetra={cells.shape[0]} boundary_faces={boundary_faces.shape[0]} "
        f"mesh_is_local={bool(args.mesh_is_local)}"
    )
    print(
        "[grid] "
        f"shape_zyx={tuple(int(v) for v in gt_mask.shape)} "
        f"spacing_xyz={spacing_xyz.astype(np.float32).tolist()} "
        f"origin_xyz={origin_xyz.astype(np.float32).tolist()}"
    )

    points = _make_grid_points_zyx(gt_mask.shape, origin_xyz=origin_xyz, spacing_xyz=spacing_xyz)
    use_torch_cuda = args.backend == "torch_cuda" or (args.backend == "auto" and torch.cuda.is_available())
    if use_torch_cuda:
        pred_flat = points_inside_closed_surface_torch(
            points,
            surface_vertices=vertices,
            surface_faces=boundary_faces,
            point_chunk_size=int(args.point_chunk_size),
            face_chunk_size=int(args.face_chunk_size),
            winding_threshold=float(args.winding_threshold),
            device="cuda",
        )
    else:
        pred_flat = points_inside_closed_surface(
            points,
            surface_vertices=vertices,
            surface_faces=boundary_faces,
            point_chunk_size=int(args.point_chunk_size),
            face_chunk_size=int(args.face_chunk_size),
            winding_threshold=float(args.winding_threshold),
        )
    pred_mask = pred_flat.reshape(gt_mask.shape)

    dice = _dice(pred_mask, gt_mask)
    summary = {
        "mesh": str(Path(args.mesh).resolve()),
        "label_nifti": str(Path(args.label_nifti).resolve()),
        "label_value": int(args.label_value),
        "mesh_is_local": bool(args.mesh_is_local),
        "target_rescale": float(args.target_rescale),
        "volume_shape_zyx": [int(v) for v in gt_mask.shape],
        "spacing_xyz": spacing_xyz.astype(np.float32).tolist(),
        "origin_xyz": origin_xyz.astype(np.float32).tolist(),
        "num_vertices": int(vertices.shape[0]),
        "num_tetrahedra": int(cells.shape[0]),
        "num_boundary_faces": int(boundary_faces.shape[0]),
        "backend": "torch_cuda" if use_torch_cuda else "numpy",
        "pred_positive_voxels": int(pred_mask.sum()),
        "gt_positive_voxels": int(gt_mask.sum()),
        "hard_dice": float(dice),
    }
    print(json.dumps(summary, indent=2))

    if args.output_dir is not None:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        np.save(output_dir / "pred_mask_geometric.npy", pred_mask.astype(np.uint8))
        np.save(output_dir / "gt_mask.npy", gt_mask.astype(np.uint8))
        _save_nifti_like(pred_mask, volume["image"], output_dir / "pred_mask_geometric.nii.gz")
        _save_nifti_like(gt_mask, volume["image"], output_dir / "gt_mask.nii.gz")
        with open(output_dir / "geometric_dice_summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
