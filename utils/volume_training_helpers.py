from __future__ import annotations

import json
import os
from argparse import Namespace
from datetime import datetime
from pathlib import Path

import meshio
import nibabel as nib
import numpy as np
import torch

from gaussian_renderer.volume_grid import binary_dice_loss_3d
from scene import HierarchicalTetrahedraModel
from scene.dataset_readers import fetchmsh
from utils.graphics_utils import BasicTetrahedra

try:
    from torch.utils.tensorboard import SummaryWriter

    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    MATPLOTLIB_FOUND = True
except ImportError:
    MATPLOTLIB_FOUND = False


def _squeeze_to_xyz(data: np.ndarray) -> np.ndarray:
    array = np.asarray(data, dtype=np.float32)
    if array.ndim == 3:
        return array
    if array.ndim == 4:
        squeezed = np.squeeze(array)
        if squeezed.ndim != 3:
            raise ValueError(f"Expected a 3D volume or singleton 4D volume, got shape {array.shape}.")
        return np.asarray(squeezed, dtype=np.float32)
    raise ValueError(f"Expected a 3D volume or singleton 4D volume, got shape {array.shape}.")


def _load_label_volume_compat(nifti_path: str) -> dict:
    image = nib.load(nifti_path)
    data_xyz = _squeeze_to_xyz(image.get_fdata(dtype=np.float32))
    array_zyx = np.transpose(data_xyz, (2, 1, 0))

    affine = np.asarray(image.affine, dtype=np.float32)
    basis = affine[:3, :3]
    spacing = np.linalg.norm(basis, axis=0).astype(np.float32)
    safe_spacing = np.where(spacing > 0, spacing, 1.0).astype(np.float32)
    direction = (basis / safe_spacing[None, :]).astype(np.float32)
    origin = affine[:3, 3].astype(np.float32)

    return {
        "image": image,
        "array": array_zyx,
        "spacing": spacing,
        "origin": origin,
        "direction": direction,
        "inv_direction": np.linalg.inv(direction).astype(np.float32),
        "size_xyz": np.array(data_xyz.shape, dtype=np.int32),
        "affine": affine,
    }


def _build_training_tetra(mesh_path: str, volume: dict, apply_inv_direction: bool = True) -> BasicTetrahedra:
    tetra = fetchmsh(mesh_path)
    local_vertices = np.asarray(tetra.vertices, dtype=np.float32)
    if apply_inv_direction:
        local_vertices = local_vertices @ np.asarray(volume["inv_direction"], dtype=np.float32).T
    return BasicTetrahedra(
        vertices=local_vertices.astype(np.float32),
        cells=np.asarray(tetra.cells, dtype=np.int32),
        colors=tetra.colors,
    )


def _get_scaled_spacing_xyz(volume: dict, scale: float = 0.01) -> np.ndarray:
    return np.asarray(volume["spacing"], dtype=np.float32) * float(scale)


def _get_centered_grid_origin_xyz(size_xyz: np.ndarray, spacing_xyz: np.ndarray) -> np.ndarray:
    size_xyz = np.asarray(size_xyz, dtype=np.float32)
    spacing_xyz = np.asarray(spacing_xyz, dtype=np.float32)
    return (-0.5 * (size_xyz - 1.0) * spacing_xyz).astype(np.float32)


def _compute_training_roi(
    training_tetra: BasicTetrahedra,
    gt_mask_zyx: np.ndarray,
    size_xyz: np.ndarray,
    spacing_xyz: np.ndarray,
    margin_units: float = 2.0,
) -> dict:
    size_xyz = np.asarray(size_xyz, dtype=np.int32)
    spacing_xyz = np.asarray(spacing_xyz, dtype=np.float32)
    full_origin_xyz = _get_centered_grid_origin_xyz(size_xyz, spacing_xyz)
    margin_xyz = np.full((3,), float(margin_units), dtype=np.float32)

    mesh_vertices = np.asarray(training_tetra.vertices, dtype=np.float32)
    mesh_bbox_min = mesh_vertices.min(axis=0) - margin_xyz
    mesh_bbox_max = mesh_vertices.max(axis=0) + margin_xyz

    positive_zyx = np.argwhere(gt_mask_zyx > 0.5)
    valid_z_start = None
    valid_z_end = None
    if positive_zyx.size > 0:
        gt_xyz = np.stack(
            [
                full_origin_xyz[0] + positive_zyx[:, 2].astype(np.float32) * spacing_xyz[0],
                full_origin_xyz[1] + positive_zyx[:, 1].astype(np.float32) * spacing_xyz[1],
                full_origin_xyz[2] + positive_zyx[:, 0].astype(np.float32) * spacing_xyz[2],
            ],
            axis=1,
        )
        gt_bbox_min = gt_xyz.min(axis=0) - margin_xyz
        gt_bbox_max = gt_xyz.max(axis=0) + margin_xyz
        valid_z_start = int(positive_zyx[:, 0].min())
        valid_z_end = int(positive_zyx[:, 0].max()) + 1
    else:
        gt_bbox_min = mesh_bbox_min.copy()
        gt_bbox_max = mesh_bbox_max.copy()

    union_bbox_min = np.minimum(gt_bbox_min, mesh_bbox_min)
    union_bbox_max = np.maximum(gt_bbox_max, mesh_bbox_max)

    start_idx_xyz = np.floor((union_bbox_min - full_origin_xyz) / spacing_xyz).astype(np.int32)
    end_idx_xyz = np.ceil((union_bbox_max - full_origin_xyz) / spacing_xyz).astype(np.int32) + 1
    start_idx_xyz = np.clip(start_idx_xyz, 0, size_xyz - 1)
    end_idx_xyz = np.clip(end_idx_xyz, start_idx_xyz + 1, size_xyz)
    if valid_z_start is not None and valid_z_end is not None:
        start_idx_xyz[2] = np.int32(np.clip(valid_z_start, 0, int(size_xyz[2]) - 1))
        end_idx_xyz[2] = np.int32(np.clip(valid_z_end, int(start_idx_xyz[2]) + 1, int(size_xyz[2])))

    roi_origin_xyz = full_origin_xyz + start_idx_xyz.astype(np.float32) * spacing_xyz
    roi_shape_xyz = end_idx_xyz - start_idx_xyz

    return {
        "full_origin_xyz": full_origin_xyz.astype(np.float32),
        "roi_origin_xyz": roi_origin_xyz.astype(np.float32),
        "start_idx_xyz": start_idx_xyz.astype(np.int32),
        "end_idx_xyz": end_idx_xyz.astype(np.int32),
        "roi_shape_xyz": roi_shape_xyz.astype(np.int32),
        "roi_shape_zyx": np.array([roi_shape_xyz[2], roi_shape_xyz[1], roi_shape_xyz[0]], dtype=np.int32),
        "start_idx_zyx": np.array([start_idx_xyz[2], start_idx_xyz[1], start_idx_xyz[0]], dtype=np.int32),
        "end_idx_zyx": np.array([end_idx_xyz[2], end_idx_xyz[1], end_idx_xyz[0]], dtype=np.int32),
        "gt_bbox_min_xyz": gt_bbox_min.astype(np.float32),
        "gt_bbox_max_xyz": gt_bbox_max.astype(np.float32),
        "mesh_bbox_min_xyz": mesh_bbox_min.astype(np.float32),
        "mesh_bbox_max_xyz": mesh_bbox_max.astype(np.float32),
        "union_bbox_min_xyz": union_bbox_min.astype(np.float32),
        "union_bbox_max_xyz": union_bbox_max.astype(np.float32),
        "margin_units": float(margin_units),
        "loss_roi_z_mode": "gt_nonempty_slices" if valid_z_start is not None else "mesh_bbox_fallback",
        "loss_roi_valid_z_start": None if valid_z_start is None else int(valid_z_start),
        "loss_roi_valid_z_end": None if valid_z_end is None else int(valid_z_end),
        "loss_roi_valid_z_count": 0 if valid_z_start is None else int(valid_z_end - valid_z_start),
    }


def _setup_full_boundary_surface_regularizer(tets: HierarchicalTetrahedraModel, lod=0):
    cells = tets.get_cells(lod).detach().cpu().numpy().astype(np.int64)
    if cells.shape[0] == 0:
        tets.boundary_face_indices = torch.empty((0, 3), dtype=torch.long, device=tets._xyz.device)
        tets.boundary_edge_indices = torch.empty((0, 2), dtype=torch.long, device=tets._xyz.device)
        tets.boundary_adjacent_face_pairs = torch.empty((0, 2), dtype=torch.long, device=tets._xyz.device)
        tets.boundary_vertex_indices = torch.empty((0,), dtype=torch.long, device=tets._xyz.device)
        tets.boundary_initial_vertices = torch.empty((0, 3), dtype=tets._xyz.dtype, device=tets._xyz.device)
        tets.boundary_reg_range = None
        return

    face_patterns = np.asarray([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]], dtype=np.int64)
    faces = cells[:, face_patterns].reshape(-1, 3)
    sorted_faces = np.sort(faces, axis=1)

    face_map = {}
    for idx, key in enumerate(map(tuple, sorted_faces.tolist())):
        if key in face_map:
            face_map[key]["count"] += 1
        else:
            face_map[key] = {"count": 1, "face": faces[idx]}

    boundary_faces_np = np.asarray(
        [entry["face"] for entry in face_map.values() if entry["count"] == 1],
        dtype=np.int64,
    )
    if boundary_faces_np.size == 0:
        tets.boundary_face_indices = torch.empty((0, 3), dtype=torch.long, device=tets._xyz.device)
        tets.boundary_edge_indices = torch.empty((0, 2), dtype=torch.long, device=tets._xyz.device)
        tets.boundary_adjacent_face_pairs = torch.empty((0, 2), dtype=torch.long, device=tets._xyz.device)
        tets.boundary_vertex_indices = torch.empty((0,), dtype=torch.long, device=tets._xyz.device)
        tets.boundary_initial_vertices = torch.empty((0, 3), dtype=tets._xyz.dtype, device=tets._xyz.device)
        tets.boundary_reg_range = None
        return

    edge_patterns = np.asarray([[0, 1], [1, 2], [2, 0]], dtype=np.int64)
    boundary_edges_np = boundary_faces_np[:, edge_patterns].reshape(-1, 2)
    boundary_edges_np = np.sort(boundary_edges_np, axis=1)
    boundary_edges_np = np.unique(boundary_edges_np, axis=0)
    boundary_vertices_np = np.unique(boundary_edges_np.reshape(-1))

    face_edge_map = {}
    for face_idx, face in enumerate(boundary_faces_np):
        for edge in face[edge_patterns]:
            key = tuple(np.sort(edge).tolist())
            face_edge_map.setdefault(key, []).append(face_idx)
    adjacent_face_pairs = []
    for face_ids in face_edge_map.values():
        if len(face_ids) < 2:
            continue
        for i in range(len(face_ids)):
            for j in range(i + 1, len(face_ids)):
                adjacent_face_pairs.append((face_ids[i], face_ids[j]))
    adjacent_face_pairs_np = np.asarray(adjacent_face_pairs, dtype=np.int64).reshape(-1, 2)

    tets.boundary_face_indices = torch.as_tensor(boundary_faces_np, dtype=torch.long, device=tets._xyz.device)
    tets.boundary_edge_indices = torch.as_tensor(boundary_edges_np, dtype=torch.long, device=tets._xyz.device)
    tets.boundary_adjacent_face_pairs = torch.as_tensor(adjacent_face_pairs_np, dtype=torch.long, device=tets._xyz.device)
    tets.boundary_vertex_indices = torch.as_tensor(boundary_vertices_np, dtype=torch.long, device=tets._xyz.device)
    tets.boundary_initial_vertices = tets.get_vertices(lod).detach().clone()
    tets.boundary_reg_range = None


def _save_nifti(array_zyx: np.ndarray, reference_image: nib.spatialimages.SpatialImage, output_path: Path, dtype: np.dtype):
    array_xyz = np.ascontiguousarray(np.transpose(array_zyx, (2, 1, 0)).astype(dtype, copy=False))
    affine_xyz = np.asarray(reference_image.affine, dtype=np.float32)
    image = nib.Nifti1Image(array_xyz, affine_xyz)
    header = image.header
    header.set_data_shape(array_xyz.shape)
    header.set_data_dtype(array_xyz.dtype)

    ref_zooms = tuple(float(v) for v in reference_image.header.get_zooms()[:3])
    if len(ref_zooms) == 3:
        header.set_zooms(ref_zooms)

    xyz_unit, t_unit = reference_image.header.get_xyzt_units()
    if xyz_unit in (None, "", "unknown"):
        xyz_unit = "mm"
    if t_unit in (None, "", "unknown"):
        header.set_xyzt_units(xyz=xyz_unit)
    else:
        header.set_xyzt_units(xyz=xyz_unit, t=t_unit)

    header["cal_min"] = float(np.min(array_xyz))
    header["cal_max"] = float(np.max(array_xyz))
    header["scl_slope"] = 1.0
    header["scl_inter"] = 0.0

    qform_affine, qform_code = reference_image.get_qform(coded=True)
    sform_affine, sform_code = reference_image.get_sform(coded=True)
    image.set_qform(affine_xyz if qform_affine is None else qform_affine, code=int(qform_code) if int(qform_code) > 0 else 1)
    image.set_sform(affine_xyz if sform_affine is None else sform_affine, code=int(sform_code) if int(sform_code) > 0 else 1)
    image.update_header()
    nib.save(image, str(output_path))


def _hard_dice_score(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    pred = pred_mask.astype(np.float32)
    gt = gt_mask.astype(np.float32)
    intersection = float((pred * gt).sum())
    denom = float(pred.sum() + gt.sum())
    return (2.0 * intersection + 1.0) / (denom + 1.0 + 1e-6)


def _export_current_tetra_mesh(tets: HierarchicalTetrahedraModel, output_path: Path, lod: int):
    vertices = tets.get_vertices(lod).detach().cpu().numpy().astype(np.float64)
    cells = tets.get_cells(lod).detach().cpu().numpy().astype(np.int32)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    meshio.Mesh(points=vertices, cells=[("tetra", cells)]).write(str(output_path), file_format="gmsh")


def _prepare_output_dir(output_root: str, args_namespace: Namespace):
    output_root = os.path.abspath(output_root)
    if getattr(args_namespace, "no_timestamp_output", False):
        model_path = output_root
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = timestamp
        run_index = 1
        while os.path.exists(os.path.join(output_root, run_dir)):
            run_dir = f"{timestamp}_{run_index:02d}"
            run_index += 1
        model_path = os.path.join(output_root, run_dir)
    os.makedirs(model_path, exist_ok=True)
    with open(os.path.join(model_path, "cfg_args"), "w", encoding="utf-8") as f:
        f.write(str(args_namespace))

    tb_writer = SummaryWriter(model_path) if TENSORBOARD_FOUND else None
    return model_path, tb_writer


def _save_loss_history_artifacts(model_path: str, loss_history: list):
    if not loss_history:
        return

    jsonl_path = os.path.join(model_path, "loss_history.jsonl")
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for item in loss_history:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    if not MATPLOTLIB_FOUND:
        return

    curve_path = os.path.join(model_path, "loss_curve.png")
    iterations = [item["iteration"] for item in loss_history]
    plt.figure(figsize=(10, 6))
    plt.plot(iterations, [item["total_loss"] for item in loss_history], label="total_loss", linewidth=1.5)
    plt.plot(iterations, [item["occ_loss"] for item in loss_history], label="occ_loss", linewidth=1.2)
    plt.plot(iterations, [item["dice_loss"] for item in loss_history], label="dice_loss", linewidth=1.2)
    plt.plot(iterations, [item["l1_loss"] for item in loss_history], label="l1_loss", linewidth=1.2)
    plt.plot(iterations, [item["quality_loss"] for item in loss_history], label="quality_loss", linewidth=1.2)
    plt.plot(iterations, [item["surface_smooth_loss"] for item in loss_history], label="surface_smooth_loss", linewidth=1.2)
    plt.plot(iterations, [item["reg_loss"] for item in loss_history], label="reg_loss", linewidth=1.2)
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.title("Volume Training Loss Curve")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(curve_path, dpi=150)
    plt.close()


def _save_final_volume_outputs(model_path: str, volume: dict, pred_occ: torch.Tensor, gt_occ: torch.Tensor, binary_thresh: float):
    pred_occ_np = pred_occ.detach().cpu().numpy().astype(np.float32)
    gt_occ_np = gt_occ.detach().cpu().numpy().astype(np.uint8)
    pred_mask_np = (pred_occ_np >= float(binary_thresh)).astype(np.uint8)

    np.save(os.path.join(model_path, "pred_occ_final.npy"), pred_occ_np)
    np.save(os.path.join(model_path, "pred_mask_final.npy"), pred_mask_np)
    np.save(os.path.join(model_path, "gt_mask.npy"), gt_occ_np)

    pred_occ_nifti = Path(model_path) / "pred_occ_final.nii.gz"
    pred_mask_nifti = Path(model_path) / "pred_mask_final.nii.gz"
    gt_mask_nifti = Path(model_path) / "gt_mask.nii.gz"
    _save_nifti(pred_occ_np, volume["image"], pred_occ_nifti, np.float32)
    _save_nifti(pred_mask_np, volume["image"], pred_mask_nifti, np.uint8)
    _save_nifti(gt_occ_np, volume["image"], gt_mask_nifti, np.uint8)

    return {
        "hard_dice_score": _hard_dice_score(pred_mask_np, gt_occ_np),
        "pred_positive_voxels_hard": int(pred_mask_np.sum()),
        "pred_positive_voxels_soft_gt0": int((pred_occ_np > 0.0).sum()),
        "gt_positive_voxels": int(gt_occ_np.sum()),
        "pred_occ_final_nifti": str(pred_occ_nifti),
        "pred_mask_final_nifti": str(pred_mask_nifti),
        "gt_mask_nifti": str(gt_mask_nifti),
    }


def _sanitize_gradients(tets: HierarchicalTetrahedraModel):
    if tets.inn_network is not None:
        for param_group in tets.optimizer.param_groups:
            for param in param_group["params"]:
                if param.grad is not None:
                    torch.nan_to_num(param.grad, nan=0.0, posinf=0.0, neginf=0.0, out=param.grad)

    if tets.inn_network is not None and tets.clip_grad_norm_inn:
        torch.nn.utils.clip_grad_norm_(tets.inn_network.parameters(), tets.max_grad_norm_inn)

    if (
        hasattr(tets, "deformation_code")
        and tets.deformation_code is not None
        and tets.deformation_code.grad is not None
        and tets.clip_grad_norm_inn
    ):
        torch.nn.utils.clip_grad_norm_([tets.deformation_code], tets.max_grad_norm_inn)


def _ensure_pipeline_compat(pipe):
    if not hasattr(pipe, "hidden_size"):
        pipe.hidden_size = int(getattr(pipe, "hidden_dim", 128))
    if not hasattr(pipe, "normalization"):
        pipe.normalization = False
    if not hasattr(pipe, "freeze"):
        pipe.freeze = False


def _compute_occ_losses(pred_occ, gt_occ, args):
    dice_loss = binary_dice_loss_3d(pred_occ, gt_occ)
    l1_loss = torch.mean(torch.abs(pred_occ - gt_occ))
    occ_loss = args.lambda_volume_dice * dice_loss + args.lambda_volume_l1 * l1_loss
    return dice_loss, l1_loss, occ_loss
