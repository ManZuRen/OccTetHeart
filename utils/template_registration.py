from __future__ import annotations

import json
from itertools import permutations, product
from pathlib import Path
from typing import Iterable

import meshio
import nibabel as nib
import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree


def tetra_cells(mesh: meshio.Mesh) -> np.ndarray:
    cells = []
    for block in mesh.cells:
        if block.type == "tetra":
            cells.append(np.asarray(block.data[:, :4], dtype=np.int64))
        elif block.type == "tetra10":
            cells.append(np.asarray(block.data[:, :4], dtype=np.int64))
    if not cells:
        raise RuntimeError("No tetra cells found in template mesh.")
    return np.vstack(cells)


def boundary_faces(tets: np.ndarray) -> np.ndarray:
    counts: dict[tuple[int, int, int], tuple[int, tuple[int, int, int]]] = {}
    for a, b, c, d in tets.astype(int):
        for face in ((a, b, c), (a, b, d), (a, c, d), (b, c, d)):
            key = tuple(sorted(face))
            if key in counts:
                count, first = counts[key]
                counts[key] = (count + 1, first)
            else:
                counts[key] = (1, face)
    return np.asarray([face for _, (count, face) in counts.items() if count == 1], dtype=np.int64)


def sample_triangles(vertices: np.ndarray, faces: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    triangles = vertices[faces]
    areas = 0.5 * np.linalg.norm(
        np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]),
        axis=1,
    )
    probs = areas / max(float(areas.sum()), 1e-12)
    ids = rng.choice(len(faces), size=n, replace=True, p=probs)
    picked = triangles[ids]
    u = rng.random(n)
    v = rng.random(n)
    flip = u + v > 1.0
    u[flip] = 1.0 - u[flip]
    v[flip] = 1.0 - v[flip]
    return picked[:, 0] + u[:, None] * (picked[:, 1] - picked[:, 0]) + v[:, None] * (picked[:, 2] - picked[:, 0])


def label_boundary_direction_points(label_nii: Path, labels: Iterable[float], target_rescale: float) -> np.ndarray:
    return label_boundary_direction_points_with_z_bounds(label_nii, labels, target_rescale)[0]


def label_boundary_direction_points_with_z_bounds(
    label_nii: Path,
    labels: Iterable[float],
    target_rescale: float,
) -> tuple[np.ndarray, dict]:
    image = nib.load(str(label_nii))
    label = image.get_fdata()
    if label.ndim == 4:
        label = np.squeeze(label, axis=-1)
    if label.ndim != 3:
        raise RuntimeError(f"Expected 3D or singleton-4D NIfTI label, got shape {label.shape}.")

    mask = np.zeros(label.shape, dtype=bool)
    for value in labels:
        mask |= np.isclose(label, float(value))
    eroded = ndimage.binary_erosion(mask, structure=ndimage.generate_binary_structure(3, 1), border_value=0)
    boundary = mask & ~eroded
    vox_xyz = np.argwhere(boundary).astype(np.float64)
    if vox_xyz.shape[0] < 100:
        raise RuntimeError(f"Too few boundary points in {label_nii}. Check --registration-labels.")

    basis = np.asarray(image.affine[:3, :3], dtype=np.float64)
    spacing = np.linalg.norm(basis, axis=0)
    direction = basis / np.where(spacing > 0, spacing, 1.0)[None, :]
    center = (np.asarray(label.shape, dtype=np.float64) - 1.0) / 2.0
    axis_points = (vox_xyz - center[None, :]) * spacing[None, :] * float(target_rescale)
    direction_points = axis_points @ direction.T
    z_min = float(direction_points[:, 2].min())
    z_max = float(direction_points[:, 2].max())
    return direction_points, {
        "label_z_min": z_min,
        "label_z_max": z_max,
        "label_z_span": float(z_max - z_min),
        "num_boundary_points": int(direction_points.shape[0]),
    }


def _pca_axes(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    center = points.mean(axis=0)
    _, _, vt = np.linalg.svd(points - center, full_matrices=False)
    axes = vt.T
    if np.linalg.det(axes) < 0:
        axes[:, -1] *= -1.0
    return center, axes


def _permutation_sign_rotations(src: np.ndarray, dst: np.ndarray):
    src_center, src_axes = _pca_axes(src)
    dst_center, dst_axes = _pca_axes(dst)
    dst_size = np.ptp(dst - dst_center, axis=0)
    for perm in permutations(range(3)):
        perm_mat = np.eye(3)[:, perm]
        for signs in product((-1.0, 1.0), repeat=3):
            sign_mat = np.diag(signs)
            rot = dst_axes @ perm_mat @ sign_mat @ src_axes.T
            if np.linalg.det(rot) < 0:
                continue
            src_rot = (src - src_center) @ rot.T
            src_size = np.ptp(src_rot, axis=0)
            valid = src_size > 1e-12
            scale = float(np.median(dst_size[valid] / src_size[valid]))
            trans = dst_center - scale * (src_center @ rot.T)
            yield rot, scale, trans


def _z_constraint_targets(z_bounds: dict, margin_fraction: float, margin_abs: float) -> dict:
    label_span = max(float(z_bounds["label_z_span"]), 1e-12)
    margin = max(float(margin_abs), label_span * float(margin_fraction))
    desired_min = float(z_bounds["label_z_min"]) - margin
    desired_max = float(z_bounds["label_z_max"]) + margin
    return {
        "label_z_min": float(z_bounds["label_z_min"]),
        "label_z_max": float(z_bounds["label_z_max"]),
        "label_z_span": float(z_bounds["label_z_span"]),
        "margin": float(margin),
        "desired_z_min": desired_min,
        "desired_z_max": desired_max,
        "desired_z_span": float(desired_max - desired_min),
    }


def _z_constraint_penalties(
    points: np.ndarray,
    targets: dict,
    span_weight: float,
    range_weight: float,
) -> dict:
    z_min = float(points[:, 2].min())
    z_max = float(points[:, 2].max())
    z_span = float(z_max - z_min)
    normalizer = max(float(targets["label_z_span"]), 1e-12)
    desired_span = float(targets["desired_z_span"])
    span_error = (z_span - desired_span) / normalizer
    low_miss = max(0.0, float(targets["desired_z_min"]) - z_min) / normalizer
    high_miss = max(0.0, float(targets["desired_z_max"]) - z_max) / normalizer
    containment_miss = (
        max(0.0, float(targets["desired_z_min"]) - z_min)
        + max(0.0, float(targets["desired_z_max"]) - z_max)
    ) / normalizer
    return {
        "mesh_z_min": z_min,
        "mesh_z_max": z_max,
        "mesh_z_span": z_span,
        "span_penalty": float(span_weight) * span_error * span_error,
        "range_penalty": float(range_weight) * (low_miss * low_miss + high_miss * high_miss),
        "range_containment_miss": float(containment_miss),
    }


def _z_constraint_score(
    z_penalty_source: np.ndarray | None,
    rot: np.ndarray,
    scale: float,
    trans: np.ndarray,
    targets: dict | None,
    span_weight: float,
    range_weight: float,
) -> float:
    if z_penalty_source is None or targets is None:
        return 0.0
    moved = (z_penalty_source @ rot.T) * scale + trans[None, :]
    penalties = _z_constraint_penalties(moved, targets, span_weight, range_weight)
    return float(penalties["span_penalty"] + penalties["range_penalty"])


def _score_transform(
    src: np.ndarray,
    dst_tree: cKDTree,
    rot: np.ndarray,
    scale: float,
    trans: np.ndarray,
    trim: float,
    z_penalty_source: np.ndarray | None = None,
    z_targets: dict | None = None,
    z_span_weight: float = 0.0,
    z_range_weight: float = 0.0,
) -> float:
    moved = (src @ rot.T) * scale + trans[None, :]
    distances, _ = dst_tree.query(moved, k=1)
    cutoff = np.quantile(distances, trim)
    score = float(distances[distances <= cutoff].mean())
    score += _z_constraint_score(z_penalty_source, rot, scale, trans, z_targets, z_span_weight, z_range_weight)
    return score


def _similarity_from_pairs(src: np.ndarray, dst: np.ndarray, scale_min: float, scale_max: float):
    src_center = src.mean(axis=0)
    dst_center = dst.mean(axis=0)
    src0 = src - src_center
    dst0 = dst - dst_center
    h = src0.T @ dst0 / max(len(src), 1)
    u, singular_values, vt = np.linalg.svd(h)
    reflect = np.eye(3)
    if np.linalg.det(vt.T @ u.T) < 0:
        reflect[-1, -1] = -1.0
    rot = vt.T @ reflect @ u.T
    denom = np.sum(src0 * src0) / max(len(src), 1)
    scale = float(np.sum(singular_values * np.diag(reflect)) / max(denom, 1e-12))
    scale = float(np.clip(scale, scale_min, scale_max))
    trans = dst_center - scale * (src_center @ rot.T)
    return rot, scale, trans


def _refine_similarity_icp(
    src: np.ndarray,
    dst: np.ndarray,
    rot: np.ndarray,
    scale: float,
    trans: np.ndarray,
    maxiter: int,
    tol: float,
    trim: float,
    scale_min: float,
    scale_max: float,
) -> tuple[np.ndarray, float, np.ndarray, int]:
    tree = cKDTree(dst)
    last_mean = None
    iterations = 0
    for iterations in range(1, maxiter + 1):
        moved = (src @ rot.T) * scale + trans[None, :]
        distances, nn_idx = tree.query(moved, k=1)
        keep = distances <= np.quantile(distances, trim)
        r_delta, s_delta, t_delta = _similarity_from_pairs(moved[keep], dst[nn_idx[keep]], scale_min, scale_max)
        rot = r_delta @ rot
        scale = float(np.clip(s_delta * scale, scale_min, scale_max))
        trans = (trans @ r_delta.T) * s_delta + t_delta
        mean_distance = float(distances[keep].mean())
        if last_mean is not None and abs(last_mean - mean_distance) < tol:
            break
        last_mean = mean_distance
    return rot, scale, trans, iterations


def choose_initial_similarity(
    coarse_src: np.ndarray,
    coarse_dst: np.ndarray,
    trim: float,
    scale_min: float,
    scale_max: float,
    z_penalty_source: np.ndarray | None = None,
    z_targets: dict | None = None,
    z_span_weight: float = 0.0,
    z_range_weight: float = 0.0,
) -> tuple[np.ndarray, float, np.ndarray]:
    tree = cKDTree(coarse_dst)
    best = None
    for rot, scale, trans in _permutation_sign_rotations(coarse_src, coarse_dst):
        scale = float(np.clip(scale, scale_min, scale_max))
        score = _score_transform(
            coarse_src,
            tree,
            rot,
            scale,
            trans,
            trim,
            z_penalty_source=z_penalty_source,
            z_targets=z_targets,
            z_span_weight=z_span_weight,
            z_range_weight=z_range_weight,
        )
        if best is None or score < best[0]:
            best = (score, rot, scale, trans)
    if best is None:
        raise RuntimeError("No valid 3D rotation initialization found.")
    return best[1], best[2], best[3]


def _apply_z_constraint_correction(
    source_nodes: np.ndarray,
    rot: np.ndarray,
    scale: float,
    trans: np.ndarray,
    targets: dict,
    scale_min: float,
    scale_max: float,
) -> tuple[np.ndarray, float, np.ndarray, dict]:
    moved = (source_nodes @ rot.T) * scale + trans[None, :]
    before = _z_constraint_penalties(moved, targets, span_weight=1.0, range_weight=1.0)
    mesh_span = max(float(before["mesh_z_span"]), 1e-12)
    desired_span = float(targets["desired_z_span"])
    if mesh_span < desired_span:
        old_scale = float(scale)
        scale = float(np.clip(scale * desired_span / mesh_span, scale_min, scale_max))
        if scale != old_scale:
            moved_center = moved.mean(axis=0)
            trans = moved_center - (source_nodes @ rot.T).mean(axis=0) * scale
            moved = (source_nodes @ rot.T) * scale + trans[None, :]

    mesh_min = float(moved[:, 2].min())
    mesh_max = float(moved[:, 2].max())
    low_shift = float(targets["desired_z_min"]) - mesh_min if mesh_min > float(targets["desired_z_min"]) else None
    high_shift = float(targets["desired_z_max"]) - mesh_max if mesh_max < float(targets["desired_z_max"]) else None
    if low_shift is not None and high_shift is not None:
        shift_z = 0.5 * (low_shift + high_shift)
    elif low_shift is not None:
        shift_z = low_shift
    elif high_shift is not None:
        shift_z = high_shift
    else:
        shift_z = 0.0
    if shift_z:
        trans = trans.copy()
        trans[2] += shift_z
        moved = (source_nodes @ rot.T) * scale + trans[None, :]

    after = _z_constraint_penalties(moved, targets, span_weight=1.0, range_weight=1.0)
    return rot, scale, trans, {"before": before, "after": after, "correction_shift_z": float(shift_z)}


def align_template_mesh_to_label(
    template_mesh: meshio.Mesh,
    template_nodes: np.ndarray,
    template_boundary_faces: np.ndarray,
    template_surface_samples: np.ndarray,
    label_nii: Path,
    out_msh: Path,
    out_transform_json: Path,
    registration_labels: Iterable[float],
    target_rescale: float,
    target_points: int,
    coarse_points: int,
    maxiter: int,
    tol: float,
    trim: float,
    scale_min: float,
    scale_max: float,
    seed: int,
    z_constraint: bool = False,
    z_margin_fraction: float = 0.05,
    z_margin_abs: float = 0.0,
    z_span_penalty_weight: float = 10.0,
    z_range_penalty_weight: float = 10.0,
) -> dict:
    rng = np.random.default_rng(seed)
    dst, z_bounds = label_boundary_direction_points_with_z_bounds(label_nii, registration_labels, target_rescale)
    z_targets = _z_constraint_targets(z_bounds, z_margin_fraction, z_margin_abs) if z_constraint else None
    if dst.shape[0] > target_points:
        dst = dst[rng.choice(dst.shape[0], size=target_points, replace=False)]

    coarse_src = template_surface_samples
    if coarse_src.shape[0] > coarse_points:
        coarse_src = coarse_src[rng.choice(coarse_src.shape[0], size=coarse_points, replace=False)]
    coarse_dst = dst
    if coarse_dst.shape[0] > coarse_points:
        coarse_dst = coarse_dst[rng.choice(coarse_dst.shape[0], size=coarse_points, replace=False)]

    rot, scale, trans = choose_initial_similarity(
        coarse_src,
        coarse_dst,
        trim,
        scale_min,
        scale_max,
        z_penalty_source=template_nodes if z_constraint else None,
        z_targets=z_targets,
        z_span_weight=z_span_penalty_weight,
        z_range_weight=z_range_penalty_weight,
    )
    rot, scale, trans, iterations = _refine_similarity_icp(
        template_surface_samples,
        dst,
        rot,
        scale,
        trans,
        maxiter,
        tol,
        trim,
        scale_min,
        scale_max,
    )
    z_constraint_summary = None
    if z_targets is not None:
        rot, scale, trans, z_constraint_summary = _apply_z_constraint_correction(
            template_nodes,
            rot,
            scale,
            trans,
            z_targets,
            scale_min,
            scale_max,
        )

    aligned_nodes = (template_nodes @ rot.T) * scale + trans[None, :]
    out_msh.parent.mkdir(parents=True, exist_ok=True)
    meshio.write(
        str(out_msh),
        meshio.Mesh(
            points=aligned_nodes,
            cells=template_mesh.cells,
            point_data=template_mesh.point_data,
            cell_data=template_mesh.cell_data,
            field_data=template_mesh.field_data,
        ),
        file_format="gmsh22",
        binary=True,
    )

    moved_src = (template_surface_samples @ rot.T) * scale + trans[None, :]
    distances, _ = cKDTree(dst).query(moved_src, k=1)
    summary = {
        "label_nifti": str(label_nii),
        "aligned_mesh": str(out_msh),
        "registration_labels": [float(v) for v in registration_labels],
        "target_rescale": float(target_rescale),
        "iterations": int(iterations),
        "icp_scale": float(scale),
        "translation": trans.astype(float).tolist(),
        "rotation": rot.astype(float).tolist(),
        "mean_nn": float(distances.mean()),
        "p90_nn": float(np.quantile(distances, 0.9)),
        "bbox_min": aligned_nodes.min(axis=0).astype(float).tolist(),
        "bbox_max": aligned_nodes.max(axis=0).astype(float).tolist(),
        "num_boundary_faces": int(template_boundary_faces.shape[0]),
        "z_constraint_enabled": bool(z_constraint),
    }
    if z_targets is not None:
        summary["z_constraint"] = {
            **z_targets,
            "span_penalty_weight": float(z_span_penalty_weight),
            "range_penalty_weight": float(z_range_penalty_weight),
            **(z_constraint_summary or {}),
        }
    out_transform_json.parent.mkdir(parents=True, exist_ok=True)
    out_transform_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


def load_template_registration_inputs(template_msh: Path, source_points: int, seed: int) -> tuple[meshio.Mesh, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    mesh = meshio.read(str(template_msh))
    nodes = np.asarray(mesh.points[:, :3], dtype=np.float64)
    faces = boundary_faces(tetra_cells(mesh))
    samples = sample_triangles(nodes, faces, source_points, rng)
    return mesh, nodes, faces, samples
