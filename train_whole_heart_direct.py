#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path

import meshio
import nibabel as nib
import numpy as np


def _script_dir() -> Path:
    return Path(__file__).resolve().parent


def _default_data_root() -> Path:
    return _script_dir()/ "whole_heart"


def _default_label_nifti(data_root: Path) -> Path:
    matches = sorted(list(data_root.glob("*.nii")) + list(data_root.glob("*.nii.gz")))
    if len(matches) != 1:
        raise RuntimeError(f"Expected exactly one NIfTI under {data_root}, found {len(matches)}.")
    return matches[0]


def _default_mesh(data_root: Path) -> Path:
    matches = sorted(data_root.glob("*.msh"))
    if len(matches) != 1:
        raise RuntimeError(f"Expected exactly one .msh under {data_root}, found {len(matches)}.")
    return matches[0]


def _spacing_xyz(image: nib.spatialimages.SpatialImage) -> np.ndarray:
    basis = np.asarray(image.affine[:3, :3], dtype=np.float64)
    spacing = np.linalg.norm(basis, axis=0)
    zooms = np.asarray(image.header.get_zooms()[:3], dtype=np.float64)
    return np.where(spacing > 0, spacing, zooms)


def _center_mesh_like_training_grid(
    mesh_path: Path,
    label_nifti: Path,
    output_path: Path,
    coordinate_scale: float,
) -> dict:
    image = nib.load(str(label_nifti))
    size_xyz = np.asarray(image.shape[:3], dtype=np.float64)
    spacing_xyz = _spacing_xyz(image)
    center_mm_xyz = 0.5 * (size_xyz - 1.0) * spacing_xyz

    mesh = meshio.read(str(mesh_path))
    points = np.asarray(mesh.points, dtype=np.float64).copy()
    original_bbox_min = points[:, :3].min(axis=0)
    original_bbox_max = points[:, :3].max(axis=0)
    points[:, :3] = (points[:, :3] - center_mm_xyz[None, :]) * float(coordinate_scale)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    centered = meshio.Mesh(
        points=points,
        cells=mesh.cells,
        point_data=mesh.point_data,
        cell_data=mesh.cell_data,
        field_data=mesh.field_data,
    )
    meshio.write(str(output_path), centered, file_format="gmsh22", binary=True)

    return {
        "input_mesh": str(mesh_path.resolve()),
        "label_nifti": str(label_nifti.resolve()),
        "centered_mesh": str(output_path.resolve()),
        "nifti_shape_xyz": [int(v) for v in size_xyz],
        "nifti_spacing_xyz_mm": spacing_xyz.astype(float).tolist(),
        "nifti_center_mm_xyz": center_mm_xyz.astype(float).tolist(),
        "coordinate_transform": "(mesh_xyz_mm - nifti_center_mm_xyz) * coordinate_scale",
        "coordinate_scale": float(coordinate_scale),
        "original_bbox_min_xyz_mm": original_bbox_min.astype(float).tolist(),
        "original_bbox_max_xyz_mm": original_bbox_max.astype(float).tolist(),
        "centered_bbox_min_xyz": points[:, :3].min(axis=0).astype(float).tolist(),
        "centered_bbox_max_xyz": points[:, :3].max(axis=0).astype(float).tolist(),
    }


def _build_train_cmd(args: argparse.Namespace, centered_mesh: Path, label_nifti: Path, output_root: Path) -> list[str]:
    train_script = _script_dir() / "train_tet_soft_volume_rebase_nvp.py"
    cmd = [
        sys.executable,
        str(train_script),
        "--mesh",
        str(centered_mesh.resolve()),
        "--mesh-is-local",
        "--label-nifti",
        str(label_nifti.resolve()),
        "--label-value",
        str(args.label_value),
        "--model-path",
        str(output_root.resolve()),
        "--iterations",
        str(args.iterations),
        "--save-every",
        str(args.save_every),
        "--checkpoint-every",
        str(args.checkpoint_every),
        "--rebase-every",
        str(args.rebase_every),
        "--alpha",
        str(args.alpha),
        "--halfspace-bias",
        str(args.halfspace_bias),
        "--boundary-thresh",
        str(args.boundary_thresh),
        "--soft-boundary-width",
        str(args.soft_boundary_width),
        "--render-mode",
        args.render_mode,
        "--global-gate-steepness",
        str(args.global_gate_steepness),
        "--binary-thresh",
        str(args.binary_thresh),
        "--lambda-volume-dice",
        str(args.lambda_volume_dice),
        "--lambda-volume-l1",
        str(args.lambda_volume_l1),
        "--lambda_surface_smooth",
        f"{args.lambda_surface_smooth:g}",
        "--block-size",
        *(str(v) for v in args.block_size),
        "--tetra-chunk-size",
        str(args.tetra_chunk_size),
    ]
    if args.no_timestamp_output:
        cmd.append("--no-timestamp-output")
    if args.save_final_volume:
        cmd.append("--save-final-volume")
    if args.extra_args.strip():
        cmd.extend(shlex.split(args.extra_args))
    return cmd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Direct MRI-TET training for D:/MRI-TET/whole_heart. The input mesh is in positive mm "
            "coordinates, so it is centered to the NIfTI training grid before calling the normal trainer."
        )
    )
    parser.add_argument("--data-root", type=Path, default=_default_data_root())
    parser.add_argument("--label-nifti", type=Path, default=None)
    parser.add_argument("--mesh", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=_script_dir() / "outputs" / "whole_heart_direct")
    parser.add_argument(
        "--coordinate-scale",
        type=float,
        default=0.01,
        help="Scale after centering. Keep 0.01 to match train_tet_soft_volume_rebase_nvp.py spacing scaling.",
    )
    parser.add_argument("--overwrite-centered-mesh", action="store_true")

    parser.add_argument("--label-value", type=int, default=-1, help="Default -1 trains all nonzero labels as one foreground.")
    parser.add_argument("--iterations", type=int, default=2000)
    parser.add_argument("--save-every", type=int, default=200)
    parser.add_argument("--checkpoint-every", type=int, default=0)
    parser.add_argument("--rebase-every", type=int, default=100)
    parser.add_argument("--alpha", type=float, default=32.0)
    parser.add_argument("--halfspace-bias", type=float, default=0.05)
    parser.add_argument("--boundary-thresh", type=float, default=1e-4)
    parser.add_argument("--soft-boundary-width", type=float, default=0.0)
    parser.add_argument("--render-mode", choices=("prob_union", "sum", "sum_clip", "max"), default="max")
    parser.add_argument("--global-gate-steepness", type=float, default=50.0)
    parser.add_argument("--binary-thresh", type=float, default=0.5)
    parser.add_argument("--lambda-volume-dice", type=float, default=1.0)
    parser.add_argument("--lambda-volume-l1", type=float, default=0.5)
    parser.add_argument("--lambda_surface_smooth", type=float, default=0.0)
    parser.add_argument("--block-size", type=int, nargs=3, default=(8, 8, 8), metavar=("Z", "Y", "X"))
    parser.add_argument("--tetra-chunk-size", type=int, default=256)
    parser.add_argument("--save-final-volume", action="store_true", default=True)
    parser.add_argument("--no-timestamp-output", action="store_true")
    parser.add_argument("--extra-args", default="")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    data_root = args.data_root.resolve()
    label_nifti = (args.label_nifti or _default_label_nifti(data_root)).resolve()
    mesh_path = (args.mesh or _default_mesh(data_root)).resolve()
    output_root = args.output_root.resolve()
    preprocessed_root = output_root / "preprocessed"
    centered_mesh = preprocessed_root / f"{mesh_path.stem}_centered_scale{args.coordinate_scale:g}.msh"
    manifest_path = preprocessed_root / "whole_heart_centering_manifest.json"

    if args.overwrite_centered_mesh or not centered_mesh.exists() or not manifest_path.exists():
        manifest = _center_mesh_like_training_grid(
            mesh_path=mesh_path,
            label_nifti=label_nifti,
            output_path=centered_mesh,
            coordinate_scale=float(args.coordinate_scale),
        )
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    else:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    train_output_root = output_root / "train"
    cmd = _build_train_cmd(args, centered_mesh, label_nifti, train_output_root)
    print(f"[whole_heart] label_nifti={label_nifti}", flush=True)
    print(f"[whole_heart] input_mesh={mesh_path}", flush=True)
    print(f"[whole_heart] centered_mesh={centered_mesh}", flush=True)
    print(f"[whole_heart] coordinate_scale={args.coordinate_scale}", flush=True)
    print(f"[whole_heart] centered_bbox_min={manifest['centered_bbox_min_xyz']}", flush=True)
    print(f"[whole_heart] centered_bbox_max={manifest['centered_bbox_max_xyz']}", flush=True)
    print("[cmd] " + " ".join(shlex.quote(part) for part in cmd), flush=True)
    if args.dry_run:
        return 0
    completed = subprocess.run(cmd, cwd=_script_dir(), check=False)
    return int(completed.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
