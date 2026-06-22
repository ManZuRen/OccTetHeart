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
from gaussian_renderer.tetra_soft_occ_grid import render_htet_soft_occ_volume_torch, render_occ_volume_from_tetrahedra
from scene import HierarchicalTetrahedraModel
from utils.volume_training_helpers import (
    TENSORBOARD_FOUND,
    _build_training_tetra,
    _compute_occ_losses,
    _compute_training_roi,
    _ensure_pipeline_compat,
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
from eval_tet_mesh import evaluate_tetra_mesh_quality


def parse_args():
    parser = ArgumentParser(
        description="Train MRI-TET with per-tetra half-space 3D occupancy rendering and periodically bake the current NVP deformation into a new tetra mesh."
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
        default=(12,12,12),
        metavar=("Z", "Y", "X"),
        help="Block size used by the 3D tetra occupancy renderer.",
    )
    parser.add_argument("--tetra-chunk-size", type=int, default=512, help="Tetrahedra processed per block chunk.")
    parser.add_argument("--boundary-thresh", type=float, default=1e-4, help="Outside-threshold used to derive soft-boundary AABB expansion.")
    parser.add_argument("--soft-boundary-width", type=float, default=0.0, help="Explicit AABB expansion width in world units. Overrides --boundary-thresh.")
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
    parser.add_argument("--flip-mesh-xy", action="store_true", help="Experiment: multiply loaded mesh x/y coordinates by -1 before training.")
    parser.add_argument(
        "--mesh-flip-signs",
        type=float,
        nargs=3,
        default=None,
        metavar=("SX", "SY", "SZ"),
        help="Experiment: multiply loaded mesh coordinates by these xyz signs/scales before training, e.g. -1 -1 1.",
    )
    parser.add_argument("--optimize-global-transform", dest="optimize_global_transform", action="store_true", default=True)
    parser.add_argument("--no-optimize-global-transform", dest="optimize_global_transform", action="store_false")
    parser.add_argument("--global-translation-lr", type=float, default=1e-3)
    parser.add_argument("--global-rotation-lr", type=float, default=1e-4)
    parser.add_argument("--global-scale-lr", type=float, default=1e-4)
    parser.add_argument("--global-scale-min", type=float, default=0.5)
    parser.add_argument("--global-scale-max", type=float, default=2.0)
    parser.add_argument("--lambda-global-translation", type=float, default=0.0)
    parser.add_argument("--lambda-global-rotation", type=float, default=0.0)
    parser.add_argument("--lambda-global-scale", type=float, default=0.0)
    parser.add_argument("--registration-labels", nargs="+", type=float, default=None, help="Label values used for template ICP. Defaults to --label-value.")
    parser.add_argument("--target-rescale", type=float, default=0.01, help="Scale applied to NIfTI spacing for registration target points.")
    parser.add_argument("--source-points", type=int, default=6000)
    parser.add_argument("--target-points", type=int, default=8000)
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


def _mesh_coordinate_flip_signs(args) -> np.ndarray:
    signs = np.ones(3, dtype=np.float32)
    if bool(args.flip_mesh_xy):
        signs[:2] *= -1.0
    if args.mesh_flip_signs is not None:
        signs *= np.asarray(args.mesh_flip_signs, dtype=np.float32)
    return signs


def _apply_mesh_coordinate_flip(args, training_tetra: BasicTetrahedra) -> BasicTetrahedra:
    signs = _mesh_coordinate_flip_signs(args)

    if np.allclose(signs, np.ones(3, dtype=np.float32)):
        return training_tetra

    vertices = np.asarray(training_tetra.vertices, dtype=np.float32) * signs[None, :]
    if float(np.prod(np.sign(signs))) < 0.0:
        print(
            "[mesh-flip] warning: an odd number of axis sign flips changes tetra signed orientation; "
            "quality/non-inversion loss may react strongly."
        )
    print(f"[mesh-flip] applied coordinate multiplier xyz={signs.tolist()}")
    return BasicTetrahedra(
        vertices=vertices.astype(np.float32),
        cells=np.asarray(training_tetra.cells, dtype=np.int32),
        colors=training_tetra.colors,
    )


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


class GlobalAnisotropicTransform(torch.nn.Module):
    def __init__(self, center_xyz: np.ndarray, scale_min: float, scale_max: float, device: str = "cuda"):
        super().__init__()
        if not (float(scale_min) > 0.0 and float(scale_min) < 1.0 and float(scale_max) > 1.0):
            raise ValueError("--global-scale-min must be in (0, 1) and --global-scale-max must be > 1.")

        self.scale_min = float(scale_min)
        self.scale_max = float(scale_max)
        center = torch.as_tensor(center_xyz, dtype=torch.float32, device=device).reshape(3)
        self.register_buffer("initial_center", center.clone())
        self.register_buffer("center", center.clone())
        self.translation = torch.nn.Parameter(torch.zeros(3, dtype=torch.float32, device=device))
        self.quaternion = torch.nn.Parameter(torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float32, device=device))

        unit_ratio = (1.0 - self.scale_min) / (self.scale_max - self.scale_min)
        unit_ratio = min(max(unit_ratio, 1e-6), 1.0 - 1e-6)
        init_raw_scale = float(np.log(unit_ratio / (1.0 - unit_ratio)))
        self.raw_scale = torch.nn.Parameter(torch.full((3,), init_raw_scale, dtype=torch.float32, device=device))

    def normalized_quaternion(self) -> torch.Tensor:
        return self.quaternion / torch.clamp_min(torch.linalg.norm(self.quaternion), 1e-8)

    def scale_xyz(self) -> torch.Tensor:
        ratio = torch.sigmoid(self.raw_scale)
        return self.scale_min + (self.scale_max - self.scale_min) * ratio

    def rotation_matrix(self) -> torch.Tensor:
        q = self.normalized_quaternion()
        w, x, y, z = q.unbind()
        two = torch.tensor(2.0, dtype=q.dtype, device=q.device)
        return torch.stack(
            [
                torch.stack([1.0 - two * (y * y + z * z), two * (x * y - z * w), two * (x * z + y * w)]),
                torch.stack([two * (x * y + z * w), 1.0 - two * (x * x + z * z), two * (y * z - x * w)]),
                torch.stack([two * (x * z - y * w), two * (y * z + x * w), 1.0 - two * (x * x + y * y)]),
            ],
            dim=0,
        )

    def forward(self, vertices: torch.Tensor) -> torch.Tensor:
        centered = vertices - self.center[None, :]
        scaled = centered * self.scale_xyz()[None, :]
        rotated = scaled @ self.rotation_matrix().T
        return rotated + self.center[None, :] + self.translation[None, :]

    @torch.no_grad()
    def reset_identity(self):
        self.translation.zero_()
        self.quaternion.copy_(torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=self.quaternion.dtype, device=self.quaternion.device))
        unit_ratio = (1.0 - self.scale_min) / (self.scale_max - self.scale_min)
        unit_ratio = min(max(unit_ratio, 1e-6), 1.0 - 1e-6)
        self.raw_scale.fill_(float(np.log(unit_ratio / (1.0 - unit_ratio))))

    @torch.no_grad()
    def bake_center(self):
        self.center.copy_(self.forward(self.center[None, :])[0])


def _create_global_transform(args, training_tetra: BasicTetrahedra) -> Optional[GlobalAnisotropicTransform]:
    if not bool(args.optimize_global_transform):
        return None
    vertices = np.asarray(training_tetra.vertices, dtype=np.float32)
    center_xyz = vertices.mean(axis=0).astype(np.float32)
    return GlobalAnisotropicTransform(
        center_xyz=center_xyz,
        scale_min=float(args.global_scale_min),
        scale_max=float(args.global_scale_max),
        device="cuda",
    )


def _add_global_transform_to_optimizer(optimizer: torch.optim.Optimizer, transform: Optional[GlobalAnisotropicTransform], args):
    if transform is None:
        return
    optimizer.add_param_group({"params": [transform.translation], "lr": float(args.global_translation_lr), "name": "global_translation"})
    optimizer.add_param_group({"params": [transform.quaternion], "lr": float(args.global_rotation_lr), "name": "global_quaternion"})
    optimizer.add_param_group({"params": [transform.raw_scale], "lr": float(args.global_scale_lr), "name": "global_raw_scale"})


def _zero_global_transform_optimizer_state(optimizer: torch.optim.Optimizer, transform: Optional[GlobalAnisotropicTransform]):
    if transform is None:
        return
    for param in (transform.translation, transform.quaternion, transform.raw_scale):
        state = optimizer.state.get(param)
        if state is None:
            continue
        for value in state.values():
            if torch.is_tensor(value):
                value.zero_()


def _decay_global_transform_lrs(optimizer: torch.optim.Optimizer, decay: float = 0.5):
    for group in optimizer.param_groups:
        if str(group.get("name", "")).startswith("global_"):
            group["lr"] = float(group["lr"]) * float(decay)


def _global_transform_lrs(optimizer: torch.optim.Optimizer) -> dict:
    return {
        str(group.get("name")): float(group["lr"])
        for group in optimizer.param_groups
        if str(group.get("name", "")).startswith("global_")
    }


def _global_transform_regularization(transform: Optional[GlobalAnisotropicTransform], args) -> torch.Tensor:
    if transform is None:
        return torch.tensor(0.0, device="cuda")

    loss = torch.tensor(0.0, dtype=transform.translation.dtype, device=transform.translation.device)
    if float(args.lambda_global_translation) > 0.0:
        loss = loss + float(args.lambda_global_translation) * torch.mean(transform.translation**2)
    if float(args.lambda_global_rotation) > 0.0:
        q = transform.normalized_quaternion()
        loss = loss + float(args.lambda_global_rotation) * torch.mean(q[1:] ** 2)
    if float(args.lambda_global_scale) > 0.0:
        loss = loss + float(args.lambda_global_scale) * torch.mean(torch.log(transform.scale_xyz()) ** 2)
    return loss


def _global_transform_summary(transform: Optional[GlobalAnisotropicTransform]) -> Optional[dict]:
    if transform is None:
        return None
    with torch.no_grad():
        return {
            "initial_center_xyz": transform.initial_center.detach().cpu().numpy().astype(np.float32).tolist(),
            "center_xyz": transform.center.detach().cpu().numpy().astype(np.float32).tolist(),
            "translation_xyz": transform.translation.detach().cpu().numpy().astype(np.float32).tolist(),
            "quaternion_wxyz": transform.normalized_quaternion().detach().cpu().numpy().astype(np.float32).tolist(),
            "scale_xyz": transform.scale_xyz().detach().cpu().numpy().astype(np.float32).tolist(),
            "scale_min": float(transform.scale_min),
            "scale_max": float(transform.scale_max),
        }


def _export_current_tetra_mesh_with_global(
    tets: HierarchicalTetrahedraModel,
    output_path: Path,
    lod: int,
    global_transform: Optional[GlobalAnisotropicTransform],
):
    with torch.no_grad():
        vertices = tets.get_vertices(lod)
        if global_transform is not None:
            vertices = global_transform(vertices)
        vertices_np = vertices.detach().cpu().numpy().astype(np.float64)
        cells_np = tets.get_cells(lod).detach().cpu().numpy().astype(np.int32)
    _export_tetra_mesh_from_arrays(vertices_np, cells_np, output_path)


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


def _forward_soft_volume_occ(args, tets, volume_shape_zyx, spacing_xyz, grid_origin_xyz, global_transform=None):
    if global_transform is not None:
        lod = tets.current_lod_depth
        vertices = global_transform(tets.get_vertices(lod))
        cells = tets.get_cells(lod)
        opacities = tets.get_opacities(lod)
        return render_occ_volume_from_tetrahedra(
            vertices=vertices,
            cells=cells,
            volume_shape_zyx=volume_shape_zyx,
            voxel_spacing_xyz=np.asarray(spacing_xyz, dtype=np.float32),
            grid_origin_xyz=np.asarray(grid_origin_xyz, dtype=np.float32),
            opacities=opacities,
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
    global_transform: Optional[GlobalAnisotropicTransform],
    gt_mask_zyx: np.ndarray,
    size_xyz: np.ndarray,
    spacing_xyz: np.ndarray,
    output_dir: Path,
    iteration: int,
):
    lod = tets.current_lod_depth
    with torch.no_grad():
        baked_global_transform = _global_transform_summary(global_transform)
        baked_vertices = tets.get_vertices(lod)
        if global_transform is not None:
            baked_vertices = global_transform(baked_vertices)
        baked_vertices = baked_vertices.detach().clone()
        baked_cells = tets.get_cells(lod).detach().cpu().numpy().astype(np.int32)
        if baked_vertices.shape != tets._xyz.shape:
            raise RuntimeError(
                f"Rebase currently expects root-vertex training only, but got baked vertices {tuple(baked_vertices.shape)} "
                f"and base xyz {tuple(tets._xyz.shape)}."
            )
        tets._xyz.data.copy_(baked_vertices)
        if global_transform is not None:
            global_transform.bake_center()
            global_transform.reset_identity()

    baked_vertices_np = baked_vertices.cpu().numpy().astype(np.float32)
    stage_path = output_dir / f"tetra_rebased_iter_{iteration:06d}.msh"
    _export_tetra_mesh_from_arrays(baked_vertices_np, baked_cells, stage_path)

    tets.reset_inn(reinit_args)
    tets.boundary_initial_vertices = tets.get_vertices(lod).detach().clone()
    _zero_global_transform_optimizer_state(tets.optimizer, global_transform)
    _decay_global_transform_lrs(tets.optimizer, decay=0.5)
    tets.optimizer.zero_grad(set_to_none=True)

    baked_tetra = _make_basic_tetra(baked_vertices_np, baked_cells, template_tetra)
    roi_info = _compute_training_roi(
        baked_tetra,
        gt_mask_zyx,
        size_xyz=size_xyz,
        spacing_xyz=spacing_xyz,
        margin_units=0.02,
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
        "baked_global_transform": baked_global_transform,
    }


def main():
    overall_start_time = time.perf_counter()
    args, pipe, opt = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("train_tet_soft_volume_rebase_nvp.py currently requires CUDA because MRI-TET model initialization uses CUDA tensors.")
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
    if args.optimize_global_transform:
        if min(args.global_translation_lr, args.global_rotation_lr, args.global_scale_lr) < 0.0:
            raise ValueError("Global transform learning rates must be non-negative.")
        if not (0.0 < args.global_scale_min < 1.0 < args.global_scale_max):
            raise ValueError("Require 0 < --global-scale-min < 1 < --global-scale-max.")

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
    training_tetra = _apply_mesh_coordinate_flip(args, training_tetra)
    roi_info = _compute_training_roi(
        training_tetra,
        gt_mask_zyx,
        size_xyz=np.asarray(volume["size_xyz"], dtype=np.int32),
        spacing_xyz=spacing_xyz,
        margin_units=0.02,
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
    global_transform = _create_global_transform(args, training_tetra)
    _add_global_transform_to_optimizer(tets.optimizer, global_transform, args)
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
    if global_transform is not None:
        print(
            "[global-transform] "
            f"center={_global_transform_summary(global_transform)['center_xyz']} "
            f"translation_lr={args.global_translation_lr:.6g} "
            f"rotation_lr={args.global_rotation_lr:.6g} "
            f"scale_lr={args.global_scale_lr:.6g} "
            f"scale_bounds=({args.global_scale_min:.4f}, {args.global_scale_max:.4f})"
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
            global_transform=global_transform,
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
                global_transform=global_transform,
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
            "surface_dihedral_loss": 0.0,
            "reg_loss": 0.0,
            "global_reg_loss": 0.0,
            "num_active_tetrahedra": int(init_pred_pkg["stats"]["num_active_tetrahedra"]),
            "num_nonempty_blocks": int(init_pred_pkg["stats"]["num_nonempty_blocks"]),
            "alpha": float(args.alpha),
            "halfspace_bias": float(args.halfspace_bias),
            "ema_total_loss": float(init_occ_loss.item()),
            "rebase_count": 0,
            "global_transform": _global_transform_summary(global_transform),
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
        #
        tets.freeze_inn()
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
            global_transform=global_transform,
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

        surface_smooth_weight = float(opt.lambda_surface_smooth) * (1.0 + 5.0*float(iteration) / float(total_iterations))
        if surface_smooth_weight > 0.0:
            surface_disp_lap_loss = tets.compute_boundary_displacement_laplacian_loss(lod=0)
            surface_coord_lap_loss = tets.compute_boundary_laplacian_loss(lod=0)
            surface_dihedral_loss = tets.compute_boundary_dihedral_preservation_loss(lod=0)
            surface_smooth_loss = surface_smooth_weight * (
                surface_disp_lap_loss
                + 0.1 * surface_coord_lap_loss
                + float(opt.lambda_surface_dihedral) * surface_dihedral_loss
            )
        else:
            surface_disp_lap_loss = torch.tensor(0.0, device="cuda")
            surface_coord_lap_loss = torch.tensor(0.0, device="cuda")
            surface_dihedral_loss = torch.tensor(0.0, device="cuda")
            surface_smooth_loss = torch.tensor(0.0, device="cuda")

        global_reg_loss = _global_transform_regularization(global_transform, args)

        loss = occ_loss + reg_loss + quality_loss + surface_smooth_loss + global_reg_loss
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
                "surface_smooth_weight": float(surface_smooth_weight),
                "surface_disp_lap_loss": float(surface_disp_lap_loss.item()),
                "surface_coord_lap_loss": float(surface_coord_lap_loss.item()),
                "surface_dihedral_loss": float(surface_dihedral_loss.item()),
                "reg_loss": float(reg_loss.item()),
                "global_reg_loss": float(global_reg_loss.item()),
                "num_active_tetrahedra": int(pred_pkg["stats"]["num_active_tetrahedra"]),
                "num_nonempty_blocks": int(pred_pkg["stats"]["num_nonempty_blocks"]),
                "alpha": float(train_render_args.alpha),
                "halfspace_bias": float(train_render_args.halfspace_bias),
                "ema_total_loss": float(ema_loss),
                "rebase_count": int(len(rebase_history)),
                "global_transform": _global_transform_summary(global_transform),
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
                tb_writer.add_scalar("soft_volume_rebase_train/surface_dihedral_loss", snapshot["surface_dihedral_loss"], iteration)
                tb_writer.add_scalar("soft_volume_rebase_train/reg_loss", snapshot["reg_loss"], iteration)
                tb_writer.add_scalar("soft_volume_rebase_train/global_reg_loss", snapshot["global_reg_loss"], iteration)
                tb_writer.add_scalar("soft_volume_rebase_train/num_active_tetrahedra", snapshot["num_active_tetrahedra"], iteration)
                tb_writer.add_scalar("soft_volume_rebase_train/num_nonempty_blocks", snapshot["num_nonempty_blocks"], iteration)
                tb_writer.add_scalar("soft_volume_rebase_train/alpha", snapshot["alpha"], iteration)
                tb_writer.add_scalar("soft_volume_rebase_train/halfspace_bias", snapshot["halfspace_bias"], iteration)
                tb_writer.add_scalar("soft_volume_rebase_train/rebase_count", snapshot["rebase_count"], iteration)
                if snapshot["global_transform"] is not None:
                    scale_xyz = snapshot["global_transform"]["scale_xyz"]
                    trans_xyz = snapshot["global_transform"]["translation_xyz"]
                    tb_writer.add_scalar("soft_volume_rebase_train/global_scale_x", scale_xyz[0], iteration)
                    tb_writer.add_scalar("soft_volume_rebase_train/global_scale_y", scale_xyz[1], iteration)
                    tb_writer.add_scalar("soft_volume_rebase_train/global_scale_z", scale_xyz[2], iteration)
                    tb_writer.add_scalar("soft_volume_rebase_train/global_translation_norm", float(np.linalg.norm(trans_xyz)), iteration)

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
                torch.save(
                    {
                        "tets": tets.capture(),
                        "iteration": iteration,
                        "global_transform": None if global_transform is None else global_transform.state_dict(),
                        "global_transform_summary": _global_transform_summary(global_transform),
                    },
                    os.path.join(model_path, f"chkpnt_{iteration}.pth"),
                )

            if iteration < total_iterations and iteration % int(args.rebase_every) == 0:
                rebase_event = _bake_current_deformation_and_reset_nvp(
                    tets=tets,
                    template_tetra=training_tetra,
                    reinit_args=reinit_args,
                    global_transform=global_transform,
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
                        "baked_global_transform": rebase_event["baked_global_transform"],
                        "global_transform_lrs": _global_transform_lrs(tets.optimizer),
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
    _export_current_tetra_mesh_with_global(tets, final_mesh_path, tets.current_lod_depth, global_transform)
    final_hard_dice_args = _final_hard_dice_render_args(args)

    with torch.no_grad():
        final_roi_pred_pkg = _forward_soft_volume_occ(
            final_hard_dice_args,
            tets,
            tuple(int(v) for v in roi_info["roi_shape_zyx"]),
            spacing_xyz,
            roi_info["roi_origin_xyz"],
            global_transform=global_transform,
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
        "mesh_flip_xy": bool(args.flip_mesh_xy),
        "mesh_flip_signs_xyz": _mesh_coordinate_flip_signs(args).astype(np.float32).tolist(),
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
        "optimize_global_transform": bool(args.optimize_global_transform),
        "global_translation_lr": float(args.global_translation_lr),
        "global_rotation_lr": float(args.global_rotation_lr),
        "global_scale_lr": float(args.global_scale_lr),
        "global_scale_min": float(args.global_scale_min),
        "global_scale_max": float(args.global_scale_max),
        "lambda_global_translation": float(args.lambda_global_translation),
        "lambda_global_rotation": float(args.lambda_global_rotation),
        "lambda_global_scale": float(args.lambda_global_scale),
        "final_global_transform": _global_transform_summary(global_transform),
        "lambda_quality_final": float(opt.lambda_quality_final),
        "lambda_surface_smooth": float(opt.lambda_surface_smooth),
        "lambda_surface_dihedral": float(opt.lambda_surface_dihedral),
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
            global_transform=global_transform,
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

    eval_json_path = Path(model_path) / "final_evaluation.json"
    try:
        final_eval = evaluate_tetra_mesh_quality(
            msh_path=final_mesh_path,
            json_out=eval_json_path,
        )
        summary["final_evaluation"] = final_eval
        summary["final_evaluation_path"] = str(eval_json_path)
        summary["final_flipped_tetrahedra_count"] = int(final_eval["negative_signed_volume_count"])
        print(
            "[eval] "
            f"MR_p05={final_eval['mean_ratio_p05']:.6g} "
            f"MR_p50={final_eval['mean_ratio_p50']:.6g} "
            f"RR_p05={final_eval['radius_ratio_p05']:.6g} "
            f"RR_p50={final_eval['radius_ratio_p50']:.6g} "
            f"RR_lt_0.2={final_eval['radius_ratio_lt_0p2_count']} "
            f"neg_vol={final_eval['negative_signed_volume_count']}",
            flush=True,
        )
        print(f"[quality] flipped_tetrahedra={final_eval['negative_signed_volume_count']}", flush=True)
    except Exception as exc:
        summary["final_evaluation_error"] = str(exc)
        print(f"[eval] warning: final evaluation failed: {exc}", flush=True)

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
