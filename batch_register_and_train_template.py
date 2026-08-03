from __future__ import annotations

import argparse
import gc
import json
import re
import shlex
import subprocess
import sys
import traceback
from dataclasses import dataclass
from argparse import Namespace
from pathlib import Path
from statistics import mean
from typing import Iterable, List, Optional

from utils.template_registration import align_template_mesh_to_label, load_template_registration_inputs


@dataclass(frozen=True)
class RegisteredCaseSpec:
    case_id: str
    case_dir: Path
    mesh_path: Path
    label_path: Path


def _strip_nii_suffix(path: Path) -> str:
    name = path.name
    if name.endswith(".nii.gz"):
        return name[:-7]
    if name.endswith(".nii"):
        return name[:-4]
    return path.stem


def _safe_case_id(label_path: Path, data_root: Path) -> str:
    rel = label_path.relative_to(data_root)
    parts = list(rel.parts[:-1]) + [_strip_nii_suffix(label_path)]
    raw = "__".join(parts)
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", raw)


def discover_label_cases(data_root: Path, label_glob: str, requested_cases: Iterable[str]) -> list[tuple[str, Path]]:
    requested = {case.strip() for case in requested_cases if case.strip()}
    labels = sorted(path for path in data_root.glob(label_glob) if path.is_file() and (path.name.endswith(".nii") or path.name.endswith(".nii.gz")))
    cases = [(_safe_case_id(path, data_root), path) for path in labels]
    if requested:
        cases = [(case_id, path) for case_id, path in cases if case_id in requested]
    if not cases:
        raise RuntimeError(f"No NIfTI labels found under {data_root} with --label-glob {label_glob!r}.")
    return cases


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Register one prepared template tetra mesh to each NIfTI label, then train OccTetHeart per case."
    )
    parser.add_argument("--template-msh", type=Path, required=True, help="Prepared template tetra mesh in OccTetHeart training units.")
    parser.add_argument("--data-root", type=Path, required=True, help="Root containing NIfTI label files.")
    parser.add_argument("--label-glob", default="**/*label.nii.gz", help="Glob below --data-root used to discover labels.")
    parser.add_argument("--output-root", type=Path, default=Path("./outputs/template_registered_batch"))
    parser.add_argument("--case", action="append", default=[], help="Run only this discovered case id. Repeatable.")

    parser.add_argument("--registration-labels", nargs="+", type=float, default=[2.0], help="Label values used for template ICP.")
    parser.add_argument("--target-rescale", type=float, default=0.01, help="Scale applied to NIfTI spacing for registration target points.")
    parser.add_argument("--source-points", type=int, default=16000)
    parser.add_argument("--target-points", type=int, default=20000)
    parser.add_argument("--coarse-points", type=int, default=5000)
    parser.add_argument("--registration-maxiter", type=int, default=120)
    parser.add_argument("--registration-tol", type=float, default=1e-7)
    parser.add_argument("--registration-trim", type=float, default=0.88)
    parser.add_argument("--scale-min", type=float, default=0.35)
    parser.add_argument("--scale-max", type=float, default=2.5)
    parser.add_argument("--registration-z-constraint", action="store_true")
    parser.add_argument("--z-margin-fraction", type=float, default=0.05)
    parser.add_argument("--z-margin-abs", type=float, default=0.0)
    parser.add_argument("--z-span-penalty-weight", type=float, default=10.0)
    parser.add_argument("--z-range-penalty-weight", type=float, default=10.0)
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
    parser.add_argument("--extra-args", default="", help="Extra args parsed by OptimizationParams/PipelineParams.")

    parser.add_argument("--overwrite-registration", action="store_true")
    parser.add_argument("--overwrite-training", action="store_true")
    parser.add_argument("--registration-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--stop-on-error", action="store_true")
    return parser.parse_args()


def _case_has_completed_run(case_output_root: Path) -> bool:
    return any(case_output_root.glob("*/training_summary.json"))


def _cleanup_cuda() -> None:
    gc.collect()
    try:
        import torch
    except ImportError:
        return
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        try:
            torch.cuda.ipc_collect()
        except RuntimeError:
            pass


def _training_batch_args(args: argparse.Namespace, training_root: Path) -> Namespace:
    return Namespace(
        output_root=training_root,
        label_value=args.label_value,
        sh_degree=args.sh_degree,
        iterations=args.iterations,
        save_every=args.save_every,
        checkpoint_every=args.checkpoint_every,
        rebase_every=args.rebase_every,
        alpha=args.alpha,
        halfspace_bias=args.halfspace_bias,
        boundary_thresh=args.boundary_thresh,
        soft_boundary_width=args.soft_boundary_width,
        render_mode=args.render_mode,
        multiply_opacity=args.multiply_opacity,
        global_gate_thresh=args.global_gate_thresh,
        global_gate_steepness=args.global_gate_steepness,
        binary_thresh=args.binary_thresh,
        lambda_volume_dice=args.lambda_volume_dice,
        lambda_volume_l1=args.lambda_volume_l1,
        block_size=tuple(args.block_size),
        tetra_chunk_size=args.tetra_chunk_size,
        save_final_volume=args.save_final_volume,
        extra_args=args.extra_args,
    )


def _single_train_cmd(args: argparse.Namespace, label_path: Path, case_training_root: Path, seed: int) -> list[str]:
    cmd = [
        sys.executable,
        "train_tet_soft_volume_rebase_nvp.py",
        "--template-msh",
        str(args.template_msh.resolve()),
        "--label-nifti",
        str(label_path.resolve()),
        "--label-value",
        str(args.label_value),
        "--model-path",
        str(case_training_root),
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
        str(args.render_mode),
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
        str(seed),
    ]
    if args.registration_z_constraint:
        cmd.append("--registration-z-constraint")
        cmd.extend(["--z-margin-fraction", str(args.z_margin_fraction)])
        cmd.extend(["--z-margin-abs", str(args.z_margin_abs)])
        cmd.extend(["--z-span-penalty-weight", str(args.z_span_penalty_weight)])
        cmd.extend(["--z-range-penalty-weight", str(args.z_range_penalty_weight)])
    if args.multiply_opacity:
        cmd.append("--multiply-opacity")
    if args.global_gate_thresh is not None:
        cmd.extend(["--global-gate-thresh", str(args.global_gate_thresh)])
    if args.save_final_volume:
        cmd.append("--save-final-volume")
    if args.extra_args.strip():
        cmd.extend(shlex.split(args.extra_args))
    return cmd


def _analysis(results: List[dict]) -> dict:
    completed = [item for item in results if item["status"] == "ok"]
    failed = [item for item in results if item["status"] == "failed"]
    skipped = [item for item in results if item["status"].startswith("skipped")]
    hard = [
        item["training_summary"].get("hard_dice_score")
        for item in completed
        if item.get("training_summary") and item["training_summary"].get("hard_dice_score") is not None
    ]
    soft = [
        item["training_summary"].get("final_soft_dice_score")
        for item in completed
        if item.get("training_summary") and item["training_summary"].get("final_soft_dice_score") is not None
    ]
    out = {
        "num_completed": len(completed),
        "num_failed": len(failed),
        "num_skipped": len(skipped),
        "failed_case_ids": [item["case_id"] for item in failed],
    }
    if hard:
        out["mean_final_hard_dice_score"] = float(mean(hard))
    if soft:
        out["mean_final_soft_dice_score"] = float(mean(soft))
    return out


def main() -> int:
    args = parse_args()
    data_root = args.data_root.resolve()
    output_root = args.output_root.resolve()
    training_root = output_root / "training"
    transform_root = output_root / "registration_transforms"
    output_root.mkdir(parents=True, exist_ok=True)

    cases = discover_label_cases(data_root, args.label_glob, args.case)
    print(f"[batch] data_root={data_root}")
    print(f"[batch] template_msh={args.template_msh.resolve()}")
    print(f"[batch] output_root={output_root}")
    print(f"[batch] num_cases={len(cases)}")
    printed_train_argv = False

    template_mesh = template_nodes = template_faces = template_samples = None
    if args.registration_only:
        template_mesh, template_nodes, template_faces, template_samples = load_template_registration_inputs(
            args.template_msh.resolve(),
            source_points=args.source_points,
            seed=args.seed,
        )
    results: list[dict] = []

    for index, (case_id, label_path) in enumerate(cases, start=1):
        transform_json = transform_root / f"{case_id}_registration.json"
        case_training_root = training_root / case_id
        aligned_msh = case_training_root / "template_aligned_for_training.msh"
        print(f"[case] ({index}/{len(cases)}) {case_id}")

        if args.dry_run:
            cmd = _single_train_cmd(args, label_path, case_training_root, seed=args.seed + index)
            results.append(
                {
                    "case_id": case_id,
                    "status": "dry_run",
                    "label_nifti": str(label_path),
                    "aligned_mesh": str(aligned_msh),
                    "training_output_root": str(case_training_root),
                    "command": cmd,
                }
            )
            if not printed_train_argv:
                print(f"[train argv] {' '.join(shlex.quote(v) for v in cmd)}")
                printed_train_argv = True
            continue

        try:
            if args.registration_only:
                if aligned_msh.exists() and transform_json.exists() and not args.overwrite_registration:
                    registration_summary = json.loads(transform_json.read_text(encoding="utf-8"))
                    print(f"[register] skip existing {aligned_msh}")
                else:
                    if template_mesh is None:
                        raise RuntimeError("Internal error: registration template was not loaded.")
                    registration_summary = align_template_mesh_to_label(
                        template_mesh=template_mesh,
                        template_nodes=template_nodes,
                        template_boundary_faces=template_faces,
                        template_surface_samples=template_samples,
                        label_nii=label_path,
                        out_msh=aligned_msh,
                        out_transform_json=transform_json,
                        registration_labels=args.registration_labels,
                        target_rescale=args.target_rescale,
                        target_points=args.target_points,
                        coarse_points=args.coarse_points,
                        maxiter=args.registration_maxiter,
                        tol=args.registration_tol,
                        trim=args.registration_trim,
                        scale_min=args.scale_min,
                        scale_max=args.scale_max,
                        seed=args.seed + index,
                        z_constraint=args.registration_z_constraint,
                        z_margin_fraction=args.z_margin_fraction,
                        z_margin_abs=args.z_margin_abs,
                        z_span_penalty_weight=args.z_span_penalty_weight,
                        z_range_penalty_weight=args.z_range_penalty_weight,
                    )
                    print(
                        f"[register] mean={registration_summary['mean_nn']:.5f} "
                        f"p90={registration_summary['p90_nn']:.5f} scale={registration_summary['icp_scale']:.6f}"
                    )
                results.append(
                    {
                        "case_id": case_id,
                        "status": "registered",
                        "label_nifti": str(label_path),
                        "registration": registration_summary,
                    }
                )
                continue

            if _case_has_completed_run(case_training_root) and not args.overwrite_training:
                print(f"[train] skip existing {case_training_root}")
                results.append(
                    {
                        "case_id": case_id,
                        "status": "skipped_existing_training",
                        "label_nifti": str(label_path),
                        "training_output_root": str(case_training_root),
                    }
                )
                continue

            cmd = _single_train_cmd(args, label_path, case_training_root, seed=args.seed + index)
            if not printed_train_argv:
                print(f"[train argv] {' '.join(shlex.quote(v) for v in cmd)}")
                printed_train_argv = True
            completed = subprocess.run(cmd, cwd=Path(__file__).resolve().parent, check=False)
            if completed.returncode != 0:
                raise RuntimeError(f"single-case training failed with exit code {completed.returncode}")

            summaries = sorted(case_training_root.glob("*/training_summary.json"), key=lambda p: p.stat().st_mtime, reverse=True)
            training_summary = json.loads(summaries[0].read_text(encoding="utf-8")) if summaries else {}
            results.append(
                {
                    "case_id": case_id,
                    "status": "ok",
                    "label_nifti": str(label_path),
                    "registration": training_summary.get("registration"),
                    "training_summary": training_summary,
                    "run_dir": str(summaries[0].parent.resolve()) if summaries else str(case_training_root),
                }
            )
            continue

        except Exception as exc:
            print(f"[failed] {case_id}: {exc}")
            results.append(
                {
                    "case_id": case_id,
                    "status": "failed",
                    "label_nifti": str(label_path),
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                }
            )
            if args.stop_on_error:
                _cleanup_cuda()
                break
        finally:
            _cleanup_cuda()

    summary = {
        "template_msh": str(args.template_msh.resolve()),
        "data_root": str(data_root),
        "label_glob": args.label_glob,
        "output_root": str(output_root),
        "registration_labels": [float(v) for v in args.registration_labels],
        "target_rescale": float(args.target_rescale),
        "registration_z_constraint": bool(args.registration_z_constraint),
        "num_cases": len(cases),
        "results": results,
        "analysis": _analysis(results),
    }
    summary_path = output_root / "template_registered_batch_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[batch] summary={summary_path}")
    print(
        f"[batch] completed={summary['analysis']['num_completed']} "
        f"failed={summary['analysis']['num_failed']} skipped={summary['analysis']['num_skipped']}"
    )
    return 1 if summary["analysis"]["num_failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
