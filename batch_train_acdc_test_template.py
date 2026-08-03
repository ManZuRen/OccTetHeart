#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import json
import re
import shlex
import subprocess
import sys
import time
import traceback
from pathlib import Path
from statistics import mean


def _script_dir() -> Path:
    return Path(__file__).resolve().parent


def _repo_root() -> Path:
    return _script_dir().parent


def _default_dataset_root() -> Path:
    return _repo_root() / "acdc_test"


def _default_template_msh() -> Path:
    return _repo_root() / "Standard_LV_4055_watertight_tet_medium.msh"


def _is_nifti(path: Path) -> bool:
    name = path.name.lower()
    return path.is_file() and (name.endswith(".nii") or name.endswith(".nii.gz"))


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


def _discover_cases(data_root: Path, label_glob: str, selected_cases: list[str]) -> list[tuple[str, Path]]:
    selected = {case.strip() for case in selected_cases if case.strip()}
    labels = sorted(path for path in data_root.glob(label_glob) if _is_nifti(path))
    cases = [(_safe_case_id(path, data_root), path) for path in labels]
    if selected:
        cases = [(case_id, path) for case_id, path in cases if case_id in selected]
    if not cases:
        raise RuntimeError(f"No NIfTI labels found under {data_root} with --label-glob {label_glob!r}.")
    return cases


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
        str(case_training_root.resolve()),
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
    if args.multiply_opacity:
        cmd.append("--multiply-opacity")
    if args.registration_z_constraint:
        cmd.append("--registration-z-constraint")
        cmd.extend(["--z-margin-fraction", str(args.z_margin_fraction)])
        cmd.extend(["--z-margin-abs", str(args.z_margin_abs)])
        cmd.extend(["--z-span-penalty-weight", str(args.z_span_penalty_weight)])
        cmd.extend(["--z-range-penalty-weight", str(args.z_range_penalty_weight)])
    if args.global_gate_thresh is not None:
        cmd.extend(["--global-gate-thresh", str(args.global_gate_thresh)])
    if args.save_final_volume:
        cmd.append("--save-final-volume")
    if args.extra_args.strip():
        cmd.extend(shlex.split(args.extra_args))
    return cmd


def _analysis(results: list[dict]) -> dict:
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
    elapsed = [
        item.get("subprocess_wall_time_sec")
        for item in completed
        if item.get("subprocess_wall_time_sec") is not None
    ]
    peak_alloc_mb = [
        item["training_summary"].get("peak_cuda_memory_allocated_mb")
        for item in completed
        if item.get("training_summary") and item["training_summary"].get("peak_cuda_memory_allocated_mb") is not None
    ]
    peak_reserved_mb = [
        item["training_summary"].get("peak_cuda_memory_reserved_mb")
        for item in completed
        if item.get("training_summary") and item["training_summary"].get("peak_cuda_memory_reserved_mb") is not None
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
    if elapsed:
        out["mean_subprocess_wall_time_sec"] = float(mean(elapsed))
    if peak_alloc_mb:
        out["mean_peak_cuda_memory_allocated_mb"] = float(mean(peak_alloc_mb))
    if peak_reserved_mb:
        out["mean_peak_cuda_memory_reserved_mb"] = float(mean(peak_reserved_mb))
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Batch train OccTetHeart on acdc_test labels. Each case directly calls "
            "train_tet_soft_volume_rebase_nvp.py with --template-msh, so the single-case "
            "training script performs template registration before training. Case-local .msh files are ignored."
        )
    )
    parser.add_argument("--template-msh", type=Path, default=_default_template_msh())
    parser.add_argument("--data-root", type=Path, default=_default_dataset_root())
    parser.add_argument("--label-glob", default="**/*_shift_corrected_sax_label.nii.gz")
    parser.add_argument("--output-root", type=Path, default=_script_dir() / "outputs" / "acdc_test_template_batch")
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
    parser.add_argument("--registration-z-constraint", action="store_true")
    parser.add_argument("--z-margin-fraction", type=float, default=0.05)
    parser.add_argument("--z-margin-abs", type=float, default=0.0)
    parser.add_argument("--z-span-penalty-weight", type=float, default=10.0)
    parser.add_argument("--z-range-penalty-weight", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--label-value", type=int, default=2)
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

    parser.add_argument("--overwrite-training", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--stop-on-error", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    data_root = args.data_root.resolve()
    output_root = args.output_root.resolve()
    training_root = output_root / "training"
    output_root.mkdir(parents=True, exist_ok=True)

    cases = _discover_cases(data_root, args.label_glob, args.case)
    print(f"[acdc_test] data_root={data_root}")
    print(f"[acdc_test] template_msh={args.template_msh.resolve()}")
    print(f"[acdc_test] output_root={output_root}")
    print(f"[acdc_test] label_glob={args.label_glob}")
    print("[acdc_test] case-local .msh files are ignored")
    print("[acdc_test] each case calls train_tet_soft_volume_rebase_nvp.py --template-msh")
    print(f"[acdc_test] num_cases={len(cases)}")

    results: list[dict] = []
    printed_train_argv = False
    for index, (case_id, label_path) in enumerate(cases, start=1):
        case_training_root = training_root / case_id
        print(f"[case] ({index}/{len(cases)}) {case_id}")
        try:
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

            if args.dry_run:
                results.append(
                    {
                        "case_id": case_id,
                        "status": "dry_run",
                        "label_nifti": str(label_path),
                        "template_msh": str(args.template_msh.resolve()),
                        "training_output_root": str(case_training_root),
                        "command": cmd,
                    }
                )
                continue

            run_start_time = time.perf_counter()
            completed = subprocess.run(cmd, cwd=_script_dir(), check=False)
            subprocess_wall_time_sec = float(time.perf_counter() - run_start_time)
            if completed.returncode != 0:
                raise RuntimeError(f"single-case training failed with exit code {completed.returncode}")

            summaries = sorted(case_training_root.glob("*/training_summary.json"), key=lambda p: p.stat().st_mtime, reverse=True)
            training_summary = json.loads(summaries[0].read_text(encoding="utf-8")) if summaries else {}
            results.append(
                {
                    "case_id": case_id,
                    "status": "ok",
                    "label_nifti": str(label_path),
                    "template_msh": str(args.template_msh.resolve()),
                    "registration": training_summary.get("registration"),
                    "subprocess_wall_time_sec": subprocess_wall_time_sec,
                    "training_summary": training_summary,
                    "run_dir": str(summaries[0].parent.resolve()) if summaries else str(case_training_root),
                }
            )
        except Exception as exc:
            print(f"[failed] {case_id}: {exc}")
            results.append(
                {
                    "case_id": case_id,
                    "status": "failed",
                    "label_nifti": str(label_path),
                    "template_msh": str(args.template_msh.resolve()),
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
        "num_cases": len(cases),
        "case_local_meshes_ignored": True,
        "single_case_training_script": str((_script_dir() / "train_tet_soft_volume_rebase_nvp.py").resolve()),
        "single_case_registration_mode": "--template-msh",
        "registration_z_constraint": bool(args.registration_z_constraint),
        "results": results,
        "analysis": _analysis(results),
    }
    summary_path = output_root / "acdc_test_template_batch_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[acdc_test] summary={summary_path}")
    print(
        f"[acdc_test] completed={summary['analysis']['num_completed']} "
        f"failed={summary['analysis']['num_failed']} skipped={summary['analysis']['num_skipped']}"
    )
    return 1 if summary["analysis"]["num_failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
