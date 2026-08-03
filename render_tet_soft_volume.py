import json
import os
from argparse import ArgumentParser

import numpy as np
import torch

from gaussian_renderer.tetra_soft_occ_grid import render_occ_volume_from_tetrahedra
from gaussian_renderer.volume_grid import binary_dice_loss_3d
from utils.volume_training_helpers import (
    _build_training_tetra,
    _compute_training_roi,
    _get_scaled_spacing_xyz,
    _load_label_volume_compat,
    _prepare_output_dir,
    _save_final_volume_outputs,
)


def parse_args():
    parser = ArgumentParser(description="Render a tetrahedral mesh into a 3D occupancy volume using half-space occupancy.")
    parser.add_argument("--mesh", type=str, required=True, help="Input tetra mesh (.msh).")
    parser.add_argument("--gt-nifti", type=str, required=True, help="GT / reference NIfTI used for grid shape, direction, decimeter-scaled spacing, and export affine.")
    parser.add_argument("--label-value", type=int, default=2, help="Foreground label value inside the GT NIfTI. Use 1 for an already-binary mask.")
    parser.add_argument("--output-dir", type=str, default="./output_soft_volume_render", help="Output root directory.")
    parser.add_argument("--physical-scale", type=float, default=0.01, help="Scale factor applied to GT NIfTI spacing to match mesh units. Use 0.01 for mm->dm.")
    parser.add_argument("--alpha", type=float, default=32.0, help="Sigmoid sharpness applied to the half-space occupancy.")
    parser.add_argument("--halfspace-bias", type=float, default=0.05, help="Positive bias added inside each half-space sigmoid, i.e. sigmoid(alpha * (h + bias)).")
    parser.add_argument(
        "--block-size",
        type=int,
        nargs=3,
        default=(8, 8, 8),
        metavar=("Z", "Y", "X"),
        help="Block size used by the 3D tetra occupancy renderer.",
    )
    parser.add_argument("--tetra-chunk-size", type=int, default=128, help="Tetrahedra processed per block chunk.")
    parser.add_argument("--boundary-thresh", type=float, default=1e-4, help="Outside-threshold used to derive soft-boundary AABB expansion.")
    parser.add_argument("--soft-boundary-width", type=float, default=None, help="Explicit AABB expansion width in world units. Overrides --boundary-thresh.")
    parser.add_argument("--binary-thresh", type=float, default=0.5, help="Threshold used for hard occupancy export.")
    parser.add_argument(
        "--mode",
        type=str,
        default="max",
        choices=("prob_union", "sum", "sum_clip", "max"),
        help="Aggregation mode used across tetrahedra after per-tetra half-space occupancy evaluation.",
    )
    parser.add_argument("--multiply-opacity", action="store_true", help="Multiply each tetra contribution by a unit opacity placeholder.")
    parser.add_argument("--render-full-volume", action="store_true", help="Render the full reference volume instead of the ROI around mesh/label.")
    return parser.parse_args()


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("render_tet_soft_volume.py currently requires CUDA because OccTetHeart tetra processing is CUDA-oriented.")
    if args.tetra_chunk_size <= 0:
        raise ValueError("--tetra-chunk-size must be positive.")
    if min(args.block_size) <= 0:
        raise ValueError("--block-size values must be positive.")

    volume = _load_label_volume_compat(args.gt_nifti)
    label_zyx = np.asarray(volume["array"], dtype=np.float32)
    if int(args.label_value) >= 0:
        gt_mask_zyx = (label_zyx == float(args.label_value)).astype(np.float32)
    else:
        gt_mask_zyx = (label_zyx > 0.0).astype(np.float32)

    spacing_xyz = _get_scaled_spacing_xyz(volume, scale=args.physical_scale)
    training_tetra = _build_training_tetra(args.mesh, volume)

    if args.render_full_volume:
        render_shape_zyx = tuple(int(v) for v in gt_mask_zyx.shape)
        grid_origin_xyz = np.asarray(
            [
                -0.5 * (float(volume["size_xyz"][0]) - 1.0) * float(spacing_xyz[0]),
                -0.5 * (float(volume["size_xyz"][1]) - 1.0) * float(spacing_xyz[1]),
                -0.5 * (float(volume["size_xyz"][2]) - 1.0) * float(spacing_xyz[2]),
            ],
            dtype=np.float32,
        )
        gt_occ = torch.from_numpy(gt_mask_zyx).cuda().unsqueeze(0)
        roi_info = None
    else:
        roi_info = _compute_training_roi(
            training_tetra,
            gt_mask_zyx,
            size_xyz=np.asarray(volume["size_xyz"], dtype=np.int32),
            spacing_xyz=spacing_xyz,
            margin_units=2.0,
        )
        z0, y0, x0 = [int(v) for v in roi_info["start_idx_zyx"]]
        z1, y1, x1 = [int(v) for v in roi_info["end_idx_zyx"]]
        gt_mask_roi_zyx = gt_mask_zyx[z0:z1, y0:y1, x0:x1]
        gt_occ = torch.from_numpy(gt_mask_roi_zyx).cuda().unsqueeze(0)
        render_shape_zyx = tuple(int(v) for v in roi_info["roi_shape_zyx"])
        grid_origin_xyz = roi_info["roi_origin_xyz"]

    vertices = torch.as_tensor(training_tetra.vertices, device="cuda", dtype=torch.float32)
    cells = torch.as_tensor(training_tetra.cells, device="cuda", dtype=torch.long)
    opacities = torch.ones((cells.shape[0], 1), device="cuda", dtype=torch.float32)

    render_pkg = render_occ_volume_from_tetrahedra(
        vertices=vertices,
        cells=cells,
        volume_shape_zyx=render_shape_zyx,
        voxel_spacing_xyz=np.asarray(spacing_xyz, dtype=np.float32),
        grid_origin_xyz=np.asarray(grid_origin_xyz, dtype=np.float32),
        opacities=opacities,
        alpha=args.alpha,
        halfspace_bias=args.halfspace_bias,
        mode=args.mode,
        multiply_opacity=args.multiply_opacity,
        block_size=tuple(args.block_size),
        tetra_chunk_size=args.tetra_chunk_size,
        boundary_thresh=args.boundary_thresh,
        soft_boundary_width=args.soft_boundary_width,
    )
    pred_occ = render_pkg["render"]
    dice_loss = binary_dice_loss_3d(pred_occ, gt_occ)
    l1_loss = torch.mean(torch.abs(pred_occ - gt_occ))

    output_dir, tb_writer = _prepare_output_dir(args.output_dir, args)
    output_dir = os.path.abspath(output_dir)

    pred_occ_to_save = pred_occ[0]
    gt_occ_to_save = gt_occ[0]
    if roi_info is not None:
        full_shape = gt_mask_zyx.shape
        pred_occ_full = torch.zeros(full_shape, device=pred_occ.device, dtype=pred_occ.dtype)
        gt_occ_full = torch.from_numpy(gt_mask_zyx).to(device=pred_occ.device, dtype=pred_occ.dtype)
        z0, y0, x0 = [int(v) for v in roi_info["start_idx_zyx"]]
        z1, y1, x1 = [int(v) for v in roi_info["end_idx_zyx"]]
        pred_occ_full[z0:z1, y0:y1, x0:x1] = pred_occ[0]
        pred_occ_to_save = pred_occ_full
        gt_occ_to_save = gt_occ_full

    final_outputs = _save_final_volume_outputs(
        output_dir,
        volume,
        pred_occ_to_save,
        gt_occ_to_save,
        args.binary_thresh,
    )

    summary = {
        "mesh": os.path.abspath(args.mesh),
        "gt_nifti": os.path.abspath(args.gt_nifti),
        "label_value": int(args.label_value),
        "render_full_volume": bool(args.render_full_volume),
        "render_shape_zyx": [int(v) for v in render_shape_zyx],
        "source_spacing_xyz_mm": np.asarray(volume["spacing"], dtype=np.float32).tolist(),
        "render_spacing_xyz": np.asarray(spacing_xyz, dtype=np.float32).tolist(),
        "grid_origin_xyz": np.asarray(grid_origin_xyz, dtype=np.float32).tolist(),
        "physical_scale": float(args.physical_scale),
        "alpha": float(args.alpha),
        "halfspace_bias": float(args.halfspace_bias),
        "block_size_zyx": [int(v) for v in args.block_size],
        "tetra_chunk_size": int(args.tetra_chunk_size),
        "boundary_thresh": float(args.boundary_thresh),
        "soft_boundary_width": None if args.soft_boundary_width is None else float(args.soft_boundary_width),
        "mode": str(args.mode),
        "multiply_opacity": bool(args.multiply_opacity),
        "soft_dice_score": float(1.0 - dice_loss.item()),
        "l1_loss": float(l1_loss.item()),
        "num_tetrahedra": int(render_pkg["stats"]["num_tetrahedra"]),
        "num_active_tetrahedra": int(render_pkg["stats"]["num_active_tetrahedra"]),
        "num_nonempty_blocks": int(render_pkg["stats"]["num_nonempty_blocks"]),
        "num_block_assignments": int(render_pkg["stats"]["num_block_assignments"]),
        "resolved_soft_boundary_width": float(render_pkg["stats"]["soft_boundary_width"]),
    }
    if roi_info is not None:
        summary["loss_roi_origin_xyz"] = roi_info["roi_origin_xyz"].astype(np.float32).tolist()
        summary["loss_roi_shape_zyx"] = roi_info["roi_shape_zyx"].astype(np.int32).tolist()
        summary["loss_roi_start_zyx"] = roi_info["start_idx_zyx"].astype(np.int32).tolist()
        summary["loss_roi_end_zyx_exclusive"] = roi_info["end_idx_zyx"].astype(np.int32).tolist()
    summary.update(final_outputs)

    with open(os.path.join(output_dir, "render_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    if tb_writer is not None:
        tb_writer.close()

    print(f"[ok] output_dir={output_dir}")
    print(f"[ok] soft_dice_score={summary['soft_dice_score']:.6f}")
    print(f"[ok] hard_dice_score={summary['hard_dice_score']:.6f}")


if __name__ == "__main__":
    main()
