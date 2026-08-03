import json
import os
import time
from argparse import ArgumentParser, Namespace, SUPPRESS
from pathlib import Path
from typing import Optional

import meshio
import numpy as np
import torch
from tqdm import tqdm

from arguments import OptimizationParams, PipelineParams
from gaussian_renderer.tetra_soft_occ_grid import render_htet_soft_occ_volume_torch
from scene import HierarchicalTetrahedraModel
from utils.volume_training_helpers import (
    TENSORBOARD_FOUND,
    _build_training_tetra,
    _compute_occ_losses,
    _compute_training_roi,
    _ensure_pipeline_compat,
    _export_current_tetra_mesh,
    _get_scaled_spacing_xyz,
    _hard_dice_score,
    _load_label_volume_compat,
    _prepare_output_dir,
    _sanitize_gradients,
    _save_final_volume_outputs,
    _save_loss_history_artifacts,
    _setup_full_boundary_surface_regularizer,
)
from utils.template_registration import align_template_mesh_to_label, load_template_registration_inputs
from utils.graphics_utils import BasicTetrahedra


def parse_args():
    parser = ArgumentParser(
        description="Train OccTetHeart with per-tetra half-space 3D occupancy rendering and periodically bake the current NVP deformation into a new tetra mesh."
    )
    parser.add_argument("--mesh", type=str, default=None, help=SUPPRESS)
    parser.add_argument("--template-msh", type=str, default=None, help="Template tetra mesh (.msh) to register to --label-nifti before training.")
    parser.add_argument("--label-nifti", type=str, required=True, help="Input label NIfTI.")
    parser.add_argument("--label-value", type=int, default=2, help="Foreground label value. Use 1 for an already-binary mask.")
    parser.add_argument("--model-path", "-m", type=str, default="./output_soft_volume_rebase", help="Experiment root directory.")
    parser.add_argument("--no-timestamp-output", action="store_true", help=SUPPRESS)
    parser.add_argument("--sh-degree", type=int, default=3, help="SH degree used by the tetra model.")
    parser.add_argument("--alpha", type=float, default=32.0, help="Sigmoid sharpness applied to the half-space occupancy.")
    parser.add_argument("--alpha-final", type=float, default=100.0, help="Final training-time sigmoid sharpness after two-stage scheduling.")
    parser.add_argument("--halfspace-bias", type=float, default=0.05, help="Positive bias added inside each half-space sigmoid, i.e. sigmoid(alpha * (h + bias)).")
    parser.add_argument("--halfspace-bias-final", type=float, default=0.01, help="Final training-time half-space bias after two-stage scheduling.")
    parser.add_argument(
        "--sharpness-schedule",
        type=str,
        default="linear",
        choices=("linear", "stage"),
        help="Training-time alpha/bias schedule after the first half: linear interpolation or stage-wise updates at rebase boundaries.",
    )
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
    parser.add_argument(
        "--render-mode",
        type=str,
        default="max",
        choices=("prob_union", "sum", "sum_clip", "max"),
        help="Aggregation mode used across tetrahedra after per-tetra half-space occupancy evaluation.",
    )
    parser.add_argument("--multiply-opacity", action="store_true", help="Multiply each tetra contribution by the model cell opacity.")
    parser.add_argument("--global-gate-thresh", type=float, default=None, help="Optional post-aggregation occupancy gate threshold.")
    parser.add_argument("--global-gate-steepness", type=float, default=50.0, help="Sigmoid steepness used for the optional global occupancy gate.")
    parser.add_argument("--binary-thresh", type=float, default=0.5, help="Threshold used for hard Dice export.")
    parser.add_argument("--lambda-volume-dice", type=float, default=1.0, help="Weight of the 3D Dice loss.")
    parser.add_argument("--lambda-volume-l1", type=float, default=0.5, help="Weight of the L1 loss on the rendered occupancy volume.")
    parser.add_argument("--save-every", type=int, default=200, help="Iterations between metric snapshots.")
    parser.add_argument("--checkpoint-every", type=int, default=1000, help="Iterations between checkpoints. Set <= 0 to disable.")
    parser.add_argument("--save-final-volume", action="store_true", help="Export final soft/hard occupancy volumes as NIfTI/NPY.")
    parser.add_argument("--rebase-every", type=int, default=100, help="Bake the current NVP deformation into the tetra base mesh every N iterations.")
    parser.add_argument("--mesh-is-local", action="store_true", help=SUPPRESS)

    parser.add_argument("--registration-labels", nargs="+", type=float, default=None, help="Label values used for template ICP. Defaults to --label-value.")
    parser.add_argument("--target-rescale", type=float, default=0.01, help="Scale applied to NIfTI spacing for registration target points.")
    parser.add_argument("--source-points", type=int, default=16000)
    parser.add_argument("--target-points", type=int, default=20000)
    parser.add_argument("--coarse-points", type=int, default=5000)
    parser.add_argument("--registration-maxiter", type=int, default=120)
    parser.add_argument("--registration-tol", type=float, default=1e-7)
    parser.add_argument("--registration-trim", type=float, default=0.88)
    parser.add_argument("--scale-min", type=float, default=0.35)
    parser.add_argument("--scale-max", type=float, default=2.5)
    parser.add_argument("--registration-z-constraint", action="store_true", help="Constrain registered mesh z span/range to slightly cover foreground label slices.")
    parser.add_argument("--z-margin-fraction", type=float, default=0.05, help="Fraction of label z span added on both sides for z-constrained registration.")
    parser.add_argument("--z-margin-abs", type=float, default=0.0, help="Absolute z margin added on both sides for z-constrained registration.")
    parser.add_argument("--z-span-penalty-weight", type=float, default=10.0)
    parser.add_argument("--z-range-penalty-weight", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--aligned-mesh-output", type=str, default=None, help="Optional path for the template-aligned mesh written before training.")

    pipe = PipelineParams(parser)
    opt = OptimizationParams(parser)
    args = parser.parse_args()
    return args, pipe.extract(args), opt.extract(args)


def _build_reinit_args(pipe, opt) -> Namespace:
    merged = {}
    merged.update(vars(pipe))
    merged.update(vars(opt))
    args = Namespace(**merged)
    _ensure_pipeline_compat(args)
    return args


def _export_tetra_mesh_from_arrays(vertices_np: np.ndarray, cells_np: np.ndarray, output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    meshio.Mesh(
        points=np.asarray(vertices_np, dtype=np.float64),
        cells=[("tetra", np.asarray(cells_np, dtype=np.int32))],
    ).write(str(output_path), file_format="gmsh")


def _make_basic_tetra(vertices_np: np.ndarray, cells_np: np.ndarray, template_tetra: BasicTetrahedra) -> BasicTetrahedra:
    return BasicTetrahedra(
        vertices=np.asarray(vertices_np, dtype=np.float32),
        cells=np.asarray(cells_np, dtype=np.int32),
        colors=template_tetra.colors,
    )


def _prepare_training_mesh(args, model_path: str) -> tuple[str, bool, Optional[dict]]:
    if args.template_msh is None:
        if args.mesh is None:
            raise ValueError("Provide either --template-msh for automatic registration or --mesh for direct training.")
        return args.mesh, not bool(args.mesh_is_local), None

    labels = args.registration_labels
    if labels is None:
        labels = [float(args.label_value)]
    out_msh = Path(args.aligned_mesh_output) if args.aligned_mesh_output else Path(model_path) / "template_aligned_for_training.msh"
    out_json = out_msh.with_suffix(".registration.json")
    template_mesh, template_nodes, template_faces, template_samples = load_template_registration_inputs(
        Path(args.template_msh).resolve(),
        source_points=int(args.source_points),
        seed=int(args.seed),
    )
    summary = align_template_mesh_to_label(
        template_mesh=template_mesh,
        template_nodes=template_nodes,
        template_boundary_faces=template_faces,
        template_surface_samples=template_samples,
        label_nii=Path(args.label_nifti).resolve(),
        out_msh=out_msh,
        out_transform_json=out_json,
        registration_labels=labels,
        target_rescale=float(args.target_rescale),
        target_points=int(args.target_points),
        coarse_points=int(args.coarse_points),
        maxiter=int(args.registration_maxiter),
        tol=float(args.registration_tol),
        trim=float(args.registration_trim),
        scale_min=float(args.scale_min),
        scale_max=float(args.scale_max),
        seed=int(args.seed),
        z_constraint=bool(args.registration_z_constraint),
        z_margin_fraction=float(args.z_margin_fraction),
        z_margin_abs=float(args.z_margin_abs),
        z_span_penalty_weight=float(args.z_span_penalty_weight),
        z_range_penalty_weight=float(args.z_range_penalty_weight),
    )
    print(
        f"[register] template={Path(args.template_msh).resolve()} aligned_mesh={out_msh} "
        f"mean={summary['mean_nn']:.5f} p90={summary['p90_nn']:.5f} scale={summary['icp_scale']:.6f}"
    )
    return str(out_msh), False, summary


def _forward_soft_volume_occ(args, tets, volume_shape_zyx, spacing_xyz, grid_origin_xyz):
    return render_htet_soft_occ_volume_torch(
        tets,
        volume_shape_zyx=volume_shape_zyx,
        voxel_spacing_xyz=np.asarray(spacing_xyz, dtype=np.float32),
        grid_origin_xyz=np.asarray(grid_origin_xyz, dtype=np.float32),
        lod=tets.current_lod_depth,
        alpha=args.alpha,
        halfspace_bias=args.halfspace_bias,
        mode=args.render_mode,
        multiply_opacity=args.multiply_opacity,
        block_size=tuple(args.block_size),
        tetra_chunk_size=args.tetra_chunk_size,
        boundary_thresh=args.boundary_thresh,
        soft_boundary_width=args.soft_boundary_width,
        global_gate_thresh=args.global_gate_thresh,
        global_gate_steepness=args.global_gate_steepness,
    )


def _final_hard_dice_render_args(args) -> Namespace:
    eval_args = Namespace(**vars(args))
    eval_args.alpha = 500.0
    eval_args.halfspace_bias = 0.005
    return eval_args


def _scheduled_train_render_args(args, iteration: int, total_iterations: int) -> Namespace:
    render_args = Namespace(**vars(args))
    hold_iterations = max(1, total_iterations // 2)
    if iteration <= hold_iterations:
        ratio = 0.0
    elif total_iterations <= hold_iterations:
        ratio = 1.0
    elif args.sharpness_schedule == "stage":
        stage_iter = ((iteration - hold_iterations - 1) // int(args.rebase_every) + 1) * int(args.rebase_every)
        ratio = min(1.0, float(stage_iter) / float(total_iterations - hold_iterations))
    else:
        ratio = float(iteration - hold_iterations) / float(total_iterations - hold_iterations)
    render_args.alpha = float(args.alpha) + ratio * (float(args.alpha_final) - float(args.alpha))
    render_args.halfspace_bias = float(args.halfspace_bias) + ratio * (
        float(args.halfspace_bias_final) - float(args.halfspace_bias)
    )
    return render_args


def _hard_dice_from_occ(pred_occ: torch.Tensor, gt_occ: torch.Tensor, binary_thresh: float) -> float:
    pred_mask = (pred_occ.detach().cpu().numpy() >= float(binary_thresh)).astype(np.uint8)
    gt_mask = gt_occ.detach().cpu().numpy().astype(np.uint8)
    return _hard_dice_score(pred_mask, gt_mask)


def _cuda_memory_summary() -> dict:
    if not torch.cuda.is_available():
        return {
            "peak_cuda_memory_allocated_bytes": None,
            "peak_cuda_memory_reserved_bytes": None,
            "peak_cuda_memory_allocated_mb": None,
            "peak_cuda_memory_reserved_mb": None,
        }

    allocated_bytes = int(torch.cuda.max_memory_allocated())
    reserved_bytes = int(torch.cuda.max_memory_reserved())
    bytes_per_mb = 1024.0 * 1024.0
    return {
        "peak_cuda_memory_allocated_bytes": allocated_bytes,
        "peak_cuda_memory_reserved_bytes": reserved_bytes,
        "peak_cuda_memory_allocated_mb": float(allocated_bytes / bytes_per_mb),
        "peak_cuda_memory_reserved_mb": float(reserved_bytes / bytes_per_mb),
    }


def _bake_current_deformation_and_reset_nvp(
    tets: HierarchicalTetrahedraModel,
    template_tetra: BasicTetrahedra,
    reinit_args: Namespace,
    gt_mask_zyx: np.ndarray,
    size_xyz: np.ndarray,
    spacing_xyz: np.ndarray,
    output_dir: Path,
    iteration: int,
):
    lod = tets.current_lod_depth
    with torch.no_grad():
        baked_vertices = tets.get_vertices(lod).detach().clone()
        baked_cells = tets.get_cells(lod).detach().cpu().numpy().astype(np.int32)
        if baked_vertices.shape != tets._xyz.shape:
            raise RuntimeError(
                f"Rebase currently expects root-vertex training only, but got baked vertices {tuple(baked_vertices.shape)} "
                f"and base xyz {tuple(tets._xyz.shape)}."
            )
        tets._xyz.data.copy_(baked_vertices)

    baked_vertices_np = baked_vertices.cpu().numpy().astype(np.float32)
    stage_path = output_dir / f"tetra_rebased_iter_{iteration:06d}.msh"
    _export_tetra_mesh_from_arrays(baked_vertices_np, baked_cells, stage_path)

    tets.reset_inn(reinit_args)
    tets.optimizer.zero_grad(set_to_none=True)

    baked_tetra = _make_basic_tetra(baked_vertices_np, baked_cells, template_tetra)
    roi_info = _compute_training_roi(
        baked_tetra,
        gt_mask_zyx,
        size_xyz=size_xyz,
        spacing_xyz=spacing_xyz,
        margin_units=2.0,
    )
    z0, y0, x0 = [int(v) for v in roi_info["start_idx_zyx"]]
    z1, y1, x1 = [int(v) for v in roi_info["end_idx_zyx"]]
    gt_mask_roi_zyx = gt_mask_zyx[z0:z1, y0:y1, x0:x1]
    gt_occ = torch.from_numpy(gt_mask_roi_zyx).cuda().unsqueeze(0)

    return {
        "iteration": int(iteration),
        "mesh_path": str(stage_path),
        "roi_info": roi_info,
        "gt_occ": gt_occ,
        "baked_tetra": baked_tetra,
    }


def main():
    overall_start_time = time.perf_counter()
    args, pipe, opt = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("train_tet_soft_volume_rebase_nvp.py currently requires CUDA because OccTetHeart model initialization uses CUDA tensors.")
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    if args.tetra_chunk_size <= 0:
        raise ValueError("--tetra-chunk-size must be positive.")
    if min(args.block_size) <= 0:
        raise ValueError("--block-size values must be positive.")
    if args.rebase_every <= 0:
        raise ValueError("--rebase-every must be positive.")
    if args.template_msh is None and args.mesh is None:
        raise ValueError("Provide either --template-msh or --mesh.")

    _ensure_pipeline_compat(pipe)
    pipe.model_type = "NVP"

    merged_args = {}
    merged_args.update(vars(args))
    merged_args.update(vars(pipe))
    merged_args.update(vars(opt))
    output_args = Namespace(**merged_args)
    model_path, tb_writer = _prepare_output_dir(args.model_path, output_args)
    model_path = os.path.abspath(model_path)
    total_iterations = int(opt.iterations)
    rebase_dir = Path(model_path) / "rebased_meshes"
    rebase_dir.mkdir(parents=True, exist_ok=True)

    volume = _load_label_volume_compat(args.label_nifti)
    training_mesh_path, apply_inv_direction, registration_summary = _prepare_training_mesh(args, model_path)
    label_zyx = np.asarray(volume["array"], dtype=np.float32)
    if int(args.label_value) >= 0:
        gt_mask_zyx = (label_zyx == float(args.label_value)).astype(np.float32)
    else:
        gt_mask_zyx = (label_zyx > 0.0).astype(np.float32)

    spacing_xyz = _get_scaled_spacing_xyz(volume, scale=0.01)
    training_tetra = _build_training_tetra(training_mesh_path, volume, apply_inv_direction=apply_inv_direction)
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

    opt.use_cell_scale = False
    tets = HierarchicalTetrahedraModel(
        args.sh_degree,
        use_cell_opacity=opt.use_cell_opacity,
        use_cell_scale=False,
        max_depth=opt.max_depth,
        optimizable_rotation=False,
    )
    tets.create_from_tetra(training_tetra, spatial_lr_scale=1.0, low_scale_init=False)
    tets.setup_model(pipe)
    tets.training_setup(opt)
    _setup_full_boundary_surface_regularizer(tets, lod=0)
    reinit_args = _build_reinit_args(pipe, opt)

    print(
        "[roi] "
        f"shape_zyx={tuple(int(v) for v in roi_info['roi_shape_zyx'])} "
        f"start_zyx={tuple(int(v) for v in roi_info['start_idx_zyx'])} "
        f"end_zyx={tuple(int(v) for v in roi_info['end_idx_zyx'])} "
        f"margin={roi_info['margin_units']:.2f} "
        f"z_mode={roi_info['loss_roi_z_mode']} "
        f"valid_z={roi_info['loss_roi_valid_z_start']}:{roi_info['loss_roi_valid_z_end']}"
    )
    print(
        "[soft-occ] "
        f"alpha={args.alpha:.4f} "
        f"alpha_final={args.alpha_final:.4f} "
        f"halfspace_bias={args.halfspace_bias:.4f} "
        f"halfspace_bias_final={args.halfspace_bias_final:.4f} "
        f"sharpness_schedule={args.sharpness_schedule} "
        f"mode={args.render_mode} "
        f"rebase_every={args.rebase_every}"
    )

    loss_history = []
    rebase_history = []
    ema_loss = 0.0
    last_render_stats = None
    with torch.no_grad():
        init_pred_pkg = _forward_soft_volume_occ(
            args,
            tets,
            tuple(int(v) for v in roi_info["roi_shape_zyx"]),
            spacing_xyz,
            roi_info["roi_origin_xyz"],
        )
        init_pred_occ = init_pred_pkg["render"]
        init_dice_loss, init_l1_loss, init_occ_loss = _compute_occ_losses(init_pred_occ, gt_occ, args)
        init_hard_dice_score_roi = _hard_dice_from_occ(init_pred_occ, gt_occ, args.binary_thresh)
        init_full_volume_hard_dice_score = None
        if args.save_final_volume:
            init_gt_occ_full = torch.from_numpy(gt_mask_zyx).cuda().unsqueeze(0)
            init_full_pred_pkg = _forward_soft_volume_occ(
                args,
                tets,
                gt_mask_zyx.shape,
                spacing_xyz,
                roi_info["full_origin_xyz"],
            )
            init_full_volume_hard_dice_score = _hard_dice_from_occ(
                init_full_pred_pkg["render"],
                init_gt_occ_full,
                args.binary_thresh,
            )
        init_snapshot = {
            "iteration": 0,
            "total_loss": float(init_occ_loss.item()),
            "occ_loss": float(init_occ_loss.item()),
            "dice_loss": float(init_dice_loss.item()),
            "soft_dice_score": float(1.0 - init_dice_loss.item()),
            "hard_dice_score_roi": float(init_hard_dice_score_roi),
            "full_volume_hard_dice_score": None
            if init_full_volume_hard_dice_score is None
            else float(init_full_volume_hard_dice_score),
            "l1_loss": float(init_l1_loss.item()),
            "quality_loss": 0.0,
            "surface_smooth_loss": 0.0,
            "reg_loss": 0.0,
            "num_active_tetrahedra": int(init_pred_pkg["stats"]["num_active_tetrahedra"]),
            "num_nonempty_blocks": int(init_pred_pkg["stats"]["num_nonempty_blocks"]),
            "alpha": float(args.alpha),
            "halfspace_bias": float(args.halfspace_bias),
            "ema_total_loss": float(init_occ_loss.item()),
            "rebase_count": 0,
        }
        loss_history.append(init_snapshot)
        ema_loss = float(init_occ_loss.item())
        last_render_stats = dict(init_pred_pkg["stats"])
        print(
            f"[init] loss={init_snapshot['total_loss']:.6f} "
            f"soft_dice={init_snapshot['soft_dice_score']:.6f} "
            f"hard_dice_roi={init_snapshot['hard_dice_score_roi']:.6f} "
            f"hard_dice_full={init_snapshot['full_volume_hard_dice_score']} "
            f"blocks={init_snapshot['num_nonempty_blocks']} "
            f"tets={init_snapshot['num_active_tetrahedra']}"
        )

    progress_bar = tqdm(range(1, total_iterations + 1), desc="Soft volume rebase training")
    training_loop_start_time = time.perf_counter()

    for iteration in progress_bar:
        tets.update_learning_rate(iteration)
        tets.alpha_ratio = 1.0
        train_render_args = _scheduled_train_render_args(args, iteration, total_iterations)

        if iteration % 1000 == 0:
            tets.oneupSHdegree()

        pred_pkg = _forward_soft_volume_occ(
            train_render_args,
            tets,
            tuple(int(v) for v in roi_info["roi_shape_zyx"]),
            spacing_xyz,
            roi_info["roi_origin_xyz"],
        )
        pred_occ = pred_pkg["render"]
        dice_loss, l1_loss, occ_loss = _compute_occ_losses(pred_occ, gt_occ, args)

        reg_loss = torch.tensor(0.0, device="cuda")
        if opt.lambda_relu_tet > 0.0:
            reg_loss = reg_loss + opt.lambda_relu_tet * tets.signed_volume_tet_loss()

        if opt.lambda_quality > 0.0:
            if iteration > opt.lambda_quality_final_iter:
                quality_loss = opt.lambda_quality_final * tets.compute_quality_loss(threshold=opt.quality_threshold)
            else:
                quality_loss = opt.lambda_quality * tets.compute_quality_loss(threshold=opt.quality_threshold)
        else:
            quality_loss = torch.tensor(0.0, device="cuda")

        if opt.lambda_surface_smooth > 0.0:
            surface_smooth_loss = opt.lambda_surface_smooth * tets.compute_boundary_laplacian_loss(lod=0)
        else:
            surface_smooth_loss = torch.tensor(0.0, device="cuda")

        loss = occ_loss + reg_loss + quality_loss + surface_smooth_loss
        loss.backward()
        _sanitize_gradients(tets)
        tets.optimizer.step()
        tets.optimizer.zero_grad(set_to_none=True)

        with torch.no_grad():
            soft_dice_score = 1.0 - float(dice_loss.item())
            ema_loss = 0.4 * float(loss.item()) + 0.6 * ema_loss
            snapshot = {
                "iteration": int(iteration),
                "total_loss": float(loss.item()),
                "occ_loss": float(occ_loss.item()),
                "dice_loss": float(dice_loss.item()),
                "soft_dice_score": float(soft_dice_score),
                "l1_loss": float(l1_loss.item()),
                "quality_loss": float(quality_loss.item()),
                "surface_smooth_loss": float(surface_smooth_loss.item()),
                "reg_loss": float(reg_loss.item()),
                "num_active_tetrahedra": int(pred_pkg["stats"]["num_active_tetrahedra"]),
                "num_nonempty_blocks": int(pred_pkg["stats"]["num_nonempty_blocks"]),
                "alpha": float(train_render_args.alpha),
                "halfspace_bias": float(train_render_args.halfspace_bias),
                "ema_total_loss": float(ema_loss),
                "rebase_count": int(len(rebase_history)),
            }
            loss_history.append(snapshot)
            last_render_stats = dict(pred_pkg["stats"])

            if tb_writer is not None:
                tb_writer.add_scalar("soft_volume_rebase_train/total_loss", snapshot["total_loss"], iteration)
                tb_writer.add_scalar("soft_volume_rebase_train/occ_loss", snapshot["occ_loss"], iteration)
                tb_writer.add_scalar("soft_volume_rebase_train/dice_loss", snapshot["dice_loss"], iteration)
                tb_writer.add_scalar("soft_volume_rebase_train/soft_dice_score", snapshot["soft_dice_score"], iteration)
                tb_writer.add_scalar("soft_volume_rebase_train/l1_loss", snapshot["l1_loss"], iteration)
                tb_writer.add_scalar("soft_volume_rebase_train/quality_loss", snapshot["quality_loss"], iteration)
                tb_writer.add_scalar("soft_volume_rebase_train/surface_smooth_loss", snapshot["surface_smooth_loss"], iteration)
                tb_writer.add_scalar("soft_volume_rebase_train/reg_loss", snapshot["reg_loss"], iteration)
                tb_writer.add_scalar("soft_volume_rebase_train/num_active_tetrahedra", snapshot["num_active_tetrahedra"], iteration)
                tb_writer.add_scalar("soft_volume_rebase_train/num_nonempty_blocks", snapshot["num_nonempty_blocks"], iteration)
                tb_writer.add_scalar("soft_volume_rebase_train/alpha", snapshot["alpha"], iteration)
                tb_writer.add_scalar("soft_volume_rebase_train/halfspace_bias", snapshot["halfspace_bias"], iteration)
                tb_writer.add_scalar("soft_volume_rebase_train/rebase_count", snapshot["rebase_count"], iteration)

            progress_bar.set_postfix(
                {
                    "Loss": f"{snapshot['ema_total_loss']:.6f}",
                    "SoftDice": f"{snapshot['soft_dice_score']:.5f}",
                    "Alpha": f"{snapshot['alpha']:.1f}",
                    "Bias": f"{snapshot['halfspace_bias']:.4f}",
                    "Blocks": snapshot["num_nonempty_blocks"],
                    "Tets": snapshot["num_active_tetrahedra"],
                    "Rebase": snapshot["rebase_count"],
                }
            )

            if args.checkpoint_every > 0 and iteration % int(args.checkpoint_every) == 0:
                torch.save((tets.capture(), iteration), os.path.join(model_path, f"chkpnt_{iteration}.pth"))

            if iteration < total_iterations and iteration % int(args.rebase_every) == 0:
                rebase_event = _bake_current_deformation_and_reset_nvp(
                    tets=tets,
                    template_tetra=training_tetra,
                    reinit_args=reinit_args,
                    gt_mask_zyx=gt_mask_zyx,
                    size_xyz=np.asarray(volume["size_xyz"], dtype=np.int32),
                    spacing_xyz=spacing_xyz,
                    output_dir=rebase_dir,
                    iteration=iteration,
                )
                roi_info = rebase_event["roi_info"]
                gt_occ = rebase_event["gt_occ"]
                training_tetra = rebase_event["baked_tetra"]
                rebase_history.append(
                    {
                        "iteration": int(iteration),
                        "mesh_path": rebase_event["mesh_path"],
                        "roi_origin_xyz": roi_info["roi_origin_xyz"].astype(np.float32).tolist(),
                        "roi_shape_zyx": roi_info["roi_shape_zyx"].astype(np.int32).tolist(),
                        "roi_z_mode": roi_info["loss_roi_z_mode"],
                        "roi_valid_z_start": roi_info["loss_roi_valid_z_start"],
                        "roi_valid_z_end": roi_info["loss_roi_valid_z_end"],
                    }
                )
                print(
                    "[rebase] "
                    f"iter={iteration} "
                    f"mesh={rebase_event['mesh_path']} "
                    f"roi_shape_zyx={tuple(int(v) for v in roi_info['roi_shape_zyx'])} "
                    f"valid_z={roi_info['loss_roi_valid_z_start']}:{roi_info['loss_roi_valid_z_end']}"
                )

    final_mesh_path = Path(model_path) / "tetra_final_soft_volume_rebase.msh"
    _export_current_tetra_mesh(tets, final_mesh_path, tets.current_lod_depth)
    final_hard_dice_args = _final_hard_dice_render_args(args)

    with torch.no_grad():
        final_roi_pred_pkg = _forward_soft_volume_occ(
            final_hard_dice_args,
            tets,
            tuple(int(v) for v in roi_info["roi_shape_zyx"]),
            spacing_xyz,
            roi_info["roi_origin_xyz"],
        )
        final_hard_dice_score_roi = _hard_dice_from_occ(final_roi_pred_pkg["render"], gt_occ, args.binary_thresh)
    initial_hard_dice_score_roi = float(loss_history[0]["hard_dice_score_roi"]) if loss_history else None
    initial_full_volume_hard_dice_score = (
        None
        if not loss_history or loss_history[0].get("full_volume_hard_dice_score") is None
        else float(loss_history[0]["full_volume_hard_dice_score"])
    )
    torch.cuda.synchronize()
    training_loop_wall_time_sec = float(time.perf_counter() - training_loop_start_time)
    total_wall_time_sec = float(time.perf_counter() - overall_start_time)
    cuda_memory_summary = _cuda_memory_summary()

    summary = {
        "mesh": os.path.abspath(training_mesh_path),
        "input_mesh": None if args.mesh is None else os.path.abspath(args.mesh),
        "template_msh": None if args.template_msh is None else os.path.abspath(args.template_msh),
        "mesh_apply_inv_direction": bool(apply_inv_direction),
        "registration": registration_summary,
        "label_nifti": os.path.abspath(args.label_nifti),
        "label_value": int(args.label_value),
        "iterations": total_iterations,
        "volume_shape_zyx": [int(v) for v in gt_mask_zyx.shape],
        "render_spacing_xyz": spacing_xyz.astype(np.float32).tolist(),
        "source_spacing_xyz_mm": np.asarray(volume["spacing"], dtype=np.float32).tolist(),
        "full_grid_origin_xyz": roi_info["full_origin_xyz"].astype(np.float32).tolist(),
        "loss_roi_origin_xyz": roi_info["roi_origin_xyz"].astype(np.float32).tolist(),
        "loss_roi_shape_zyx": roi_info["roi_shape_zyx"].astype(np.int32).tolist(),
        "loss_roi_start_zyx": roi_info["start_idx_zyx"].astype(np.int32).tolist(),
        "loss_roi_end_zyx_exclusive": roi_info["end_idx_zyx"].astype(np.int32).tolist(),
        "loss_roi_union_bbox_min_xyz": roi_info["union_bbox_min_xyz"].astype(np.float32).tolist(),
        "loss_roi_union_bbox_max_xyz": roi_info["union_bbox_max_xyz"].astype(np.float32).tolist(),
        "loss_roi_gt_bbox_min_xyz": roi_info["gt_bbox_min_xyz"].astype(np.float32).tolist(),
        "loss_roi_gt_bbox_max_xyz": roi_info["gt_bbox_max_xyz"].astype(np.float32).tolist(),
        "loss_roi_mesh_bbox_min_xyz": roi_info["mesh_bbox_min_xyz"].astype(np.float32).tolist(),
        "loss_roi_mesh_bbox_max_xyz": roi_info["mesh_bbox_max_xyz"].astype(np.float32).tolist(),
        "loss_roi_margin_units": float(roi_info["margin_units"]),
        "loss_roi_z_mode": str(roi_info["loss_roi_z_mode"]),
        "loss_roi_valid_z_start": roi_info["loss_roi_valid_z_start"],
        "loss_roi_valid_z_end": roi_info["loss_roi_valid_z_end"],
        "loss_roi_valid_z_count": int(roi_info["loss_roi_valid_z_count"]),
        "block_size_zyx": [int(v) for v in args.block_size],
        "tetra_chunk_size": int(args.tetra_chunk_size),
        "alpha": float(args.alpha),
        "alpha_final": float(args.alpha_final),
        "halfspace_bias": float(args.halfspace_bias),
        "halfspace_bias_final": float(args.halfspace_bias_final),
        "sharpness_schedule": str(args.sharpness_schedule),
        "alpha_schedule": f"hold_first_half_then_{args.sharpness_schedule}",
        "halfspace_bias_schedule": f"hold_first_half_then_{args.sharpness_schedule}",
        "final_hard_dice_alpha": float(final_hard_dice_args.alpha),
        "final_hard_dice_halfspace_bias": float(final_hard_dice_args.halfspace_bias),
        "boundary_thresh": float(args.boundary_thresh),
        "soft_boundary_width": None if args.soft_boundary_width is None else float(args.soft_boundary_width),
        "resolved_soft_boundary_width": None if last_render_stats is None else float(last_render_stats["soft_boundary_width"]),
        "render_mode": str(args.render_mode),
        "multiply_opacity": bool(args.multiply_opacity),
        "global_gate_thresh": None if args.global_gate_thresh is None else float(args.global_gate_thresh),
        "global_gate_steepness": float(args.global_gate_steepness),
        "binary_thresh": float(args.binary_thresh),
        "lambda_volume_dice": float(args.lambda_volume_dice),
        "lambda_volume_l1": float(args.lambda_volume_l1),
        "rebase_every": int(args.rebase_every),
        "rebase_count": int(len(rebase_history)),
        "rebase_history": rebase_history,
        "training_loop_wall_time_sec": training_loop_wall_time_sec,
        "total_wall_time_sec": total_wall_time_sec,
        "final_total_loss": float(loss_history[-1]["total_loss"]) if loss_history else None,
        "final_occ_loss": float(loss_history[-1]["occ_loss"]) if loss_history else None,
        "final_soft_dice_score": float(loss_history[-1]["soft_dice_score"]) if loss_history else None,
        "initial_hard_dice_score_roi": initial_hard_dice_score_roi,
        "final_hard_dice_score_roi": float(final_hard_dice_score_roi),
        "hard_dice_improvement_roi": None
        if initial_hard_dice_score_roi is None
        else float(final_hard_dice_score_roi - initial_hard_dice_score_roi),
        "initial_full_volume_hard_dice_score": initial_full_volume_hard_dice_score,
        "final_num_tetrahedra": None if last_render_stats is None else int(last_render_stats["num_tetrahedra"]),
        "final_num_active_tetrahedra": None if last_render_stats is None else int(last_render_stats["num_active_tetrahedra"]),
        "final_num_nonempty_blocks": None if last_render_stats is None else int(last_render_stats["num_nonempty_blocks"]),
        "tetra_final_path": str(final_mesh_path),
    }
    summary.update(cuda_memory_summary)

    if args.save_final_volume:
        gt_occ_full = torch.from_numpy(gt_mask_zyx).cuda().unsqueeze(0)
        final_pred_pkg = _forward_soft_volume_occ(
            final_hard_dice_args,
            tets,
            gt_mask_zyx.shape,
            spacing_xyz,
            roi_info["full_origin_xyz"],
        )
        final_outputs = _save_final_volume_outputs(
            model_path,
            volume,
            final_pred_pkg["render"][0],
            gt_occ_full[0],
            args.binary_thresh,
        )
        summary.update(final_outputs)
        summary["final_num_tetrahedra"] = int(final_pred_pkg["stats"]["num_tetrahedra"])
        summary["final_num_active_tetrahedra"] = int(final_pred_pkg["stats"]["num_active_tetrahedra"])
        summary["final_num_nonempty_blocks"] = int(final_pred_pkg["stats"]["num_nonempty_blocks"])
        summary["resolved_soft_boundary_width"] = float(final_pred_pkg["stats"]["soft_boundary_width"])
        if summary.get("initial_full_volume_hard_dice_score") is not None:
            summary["full_volume_hard_dice_improvement"] = float(
                summary["hard_dice_score"] - summary["initial_full_volume_hard_dice_score"]
            )

    with open(os.path.join(model_path, "training_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    _save_loss_history_artifacts(model_path, loss_history)
    if tb_writer is not None:
        tb_writer.close()

    print(f"[ok] output_dir={model_path}")
    print(f"[ok] tetra_final={final_mesh_path}")
    if summary.get("final_soft_dice_score") is not None:
        print(f"[ok] final_soft_dice_score={summary['final_soft_dice_score']:.6f}")
    if summary.get("final_hard_dice_score_roi") is not None:
        print(f"[ok] final_hard_dice_score_roi={summary['final_hard_dice_score_roi']:.6f}")
    if summary.get("hard_dice_score") is not None:
        print(f"[ok] final_hard_dice_score={summary['hard_dice_score']:.6f}")
    print(f"[ok] rebase_count={len(rebase_history)}")
    print(f"[ok] training_loop_wall_time_sec={training_loop_wall_time_sec:.3f}")
    print(f"[ok] total_wall_time_sec={total_wall_time_sec:.3f}")
    if cuda_memory_summary["peak_cuda_memory_allocated_mb"] is not None:
        print(f"[ok] peak_cuda_memory_allocated_mb={cuda_memory_summary['peak_cuda_memory_allocated_mb']:.2f}")
    if cuda_memory_summary["peak_cuda_memory_reserved_mb"] is not None:
        print(f"[ok] peak_cuda_memory_reserved_mb={cuda_memory_summary['peak_cuda_memory_reserved_mb']:.2f}")


if __name__ == "__main__":
    main()
