#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path


def _script_dir() -> Path:
    return Path(__file__).resolve().parent


def _default_dataset_root() -> Path:
    return _script_dir() / "CMRxMotion"


def _default_template_msh() -> Path:
    return _script_dir() / "biv_template_generation" / "heart_sur_medium_uniform_tet32k_labeled.msh"


def _build_cmd(args: argparse.Namespace, passthrough: list[str]) -> list[str]:
    script_dir = _script_dir()
    batch_script = script_dir / "batch_register_and_train_template.py"
    cmd = [
        sys.executable,
        str(batch_script),
        "--template-msh",
        str(args.template_msh.resolve()),
        "--data-root",
        str(args.data_root.resolve()),
        "--label-glob",
        args.label_glob,
        "--output-root",
        str(args.output_root.resolve()),
        "--registration-labels",
        *(str(v) for v in args.registration_labels),
        "--target-rescale",
        str(args.target_rescale),
        "--source-points",
        str(args.source_points),
        "--target-points",
        str(args.target_points),
        "--coarse-points",
        str(args.coarse_points),
        "--registration-maxiter",
        str(args.registration_maxiter),
        "--registration-tol",
        str(args.registration_tol),
        "--registration-trim",
        str(args.registration_trim),
        "--scale-min",
        str(args.scale_min),
        "--scale-max",
        str(args.scale_max),
        "--seed",
        str(args.seed),
        "--label-value",
        str(args.label_value),
        "--sh-degree",
        str(args.sh_degree),
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
        "--block-size",
        *(str(v) for v in args.block_size),
        "--tetra-chunk-size",
        str(args.tetra_chunk_size),
    ]
    for case_id in args.case:
        cmd.extend(["--case", case_id])
    if args.multiply_opacity:
        cmd.append("--multiply-opacity")
    if args.global_gate_thresh is not None:
        cmd.extend(["--global-gate-thresh", str(args.global_gate_thresh)])
    if args.save_final_volume:
        cmd.append("--save-final-volume")
    if args.extra_args:
        cmd.extend(["--extra-args", args.extra_args])
    if args.overwrite_registration:
        cmd.append("--overwrite-registration")
    if args.overwrite_training:
        cmd.append("--overwrite-training")
    if args.registration_only:
        cmd.append("--registration-only")
    if args.dry_run:
        cmd.append("--dry-run")
    if args.stop_on_error:
        cmd.append("--stop-on-error")
    cmd.extend(passthrough)
    return cmd


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        description=(
            "Batch train MRI-TET on CMRxMotion ED/ES labels by registering one prepared "
            "template tetra mesh to each label volume."
        )
    )
    parser.add_argument("--template-msh", type=Path, default=_default_template_msh())
    parser.add_argument("--data-root", type=Path, default=_default_dataset_root())
    parser.add_argument("--label-glob", default="**/*-ED-label.nii.gz")
    parser.add_argument("--output-root", type=Path, default=_script_dir() / "outputs" / "cmrxmotion_template_batch")
    parser.add_argument("--case", action="append", default=[], help="Discovered case id. Repeatable.")

    parser.add_argument("--registration-labels", nargs="+", type=float, default=[2.0])
    parser.add_argument("--target-rescale", type=float, default=0.01)
    parser.add_argument("--source-points", type=int, default=16000)
    parser.add_argument("--target-points", type=int, default=20000)
    parser.add_argument("--coarse-points", type=int, default=5000)
    parser.add_argument("--registration-maxiter", type=int, default=120)
    parser.add_argument("--registration-tol", type=float, default=1e-7)
    parser.add_argument("--registration-trim", type=float, default=0.88)
    parser.add_argument("--scale-min", type=float, default=0.35)
    parser.add_argument("--scale-max", type=float, default=2.5)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--label-value", type=int, default=2)
    parser.add_argument("--sh-degree", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=2000)
    parser.add_argument("--save-every", type=int, default=200)
    parser.add_argument("--checkpoint-every", type=int, default=0)
    parser.add_argument("--rebase-every", type=int, default=100)
    parser.add_argument("--alpha", type=float, default=32.0)
    parser.add_argument("--halfspace-bias", type=float, default=0.05)
    parser.add_argument("--boundary-thresh", type=float, default=1e-4)
    parser.add_argument("--soft-boundary-width", type=float, default=0.0)
    parser.add_argument("--render-mode", choices=("prob_union", "sum", "sum_clip", "max"), default="max")
    parser.add_argument("--multiply-opacity", action="store_true")
    parser.add_argument("--global-gate-thresh", type=float, default=None)
    parser.add_argument("--global-gate-steepness", type=float, default=50.0)
    parser.add_argument("--binary-thresh", type=float, default=0.5)
    parser.add_argument("--lambda-volume-dice", type=float, default=1.0)
    parser.add_argument("--lambda-volume-l1", type=float, default=0.5)
    parser.add_argument("--block-size", type=int, nargs=3, default=(8, 8, 8), metavar=("Z", "Y", "X"))
    parser.add_argument("--tetra-chunk-size", type=int, default=256)
    parser.add_argument("--save-final-volume", action="store_true", default=True)
    parser.add_argument("--extra-args", default="")

    parser.add_argument("--overwrite-registration", action="store_true")
    parser.add_argument("--overwrite-training", action="store_true")
    parser.add_argument("--registration-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--stop-on-error", action="store_true")
    return parser.parse_known_args()


def main() -> int:
    args, passthrough = parse_args()
    cmd = _build_cmd(args, passthrough)
    print(f"[cmrxmotion] data_root={args.data_root.resolve()}", flush=True)
    print(f"[cmrxmotion] template_msh={args.template_msh.resolve()}", flush=True)
    print(f"[cmrxmotion] output_root={args.output_root.resolve()}", flush=True)
    completed = subprocess.run(cmd, cwd=_script_dir(), check=False)
    return int(completed.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
