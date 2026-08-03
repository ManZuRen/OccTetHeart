import argparse
import json
import math
import os
import statistics
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


LABEL_SUFFIX = "_shift_corrected_sax_label.nii.gz"
MESH_SUFFIX = "_shift_corrected_sax_label_tet_icp_aligned.msh"
CASE_SUMMARY_NAME = "occtetheart_case_summary.json"
AGGREGATE_SUMMARY_NAME = "occtetheart_acdc_testing_ghd_global_clean_batch_summary.json"


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Batch-train clean global OccTetHeart on ACDC testing GHD folders. "
            "Each patient folder should contain *_shift_corrected_sax_label.nii.gz "
            "and matching *_shift_corrected_sax_label_tet_icp_aligned.msh files."
        )
    )
    parser.add_argument("--data-root", required=True, help="Root folder containing patient subfolders.")
    parser.add_argument(
        "--output-root",
        default=None,
        help="Output root. Defaults to output/acdc_testing_ghd_global_clean_batch_<timestamp>.",
    )
    parser.add_argument("--train-script", default="train_tet_soft_volume_rebase_nvp_global_clean.py", help="OccTetHeart training script.")
    parser.add_argument("--python", default=sys.executable, help="Python executable used to launch each training job.")
    parser.add_argument("--label-value", type=int, default=2, help="Foreground label value for myocardium.")
    parser.add_argument("--iterations", type=int, default=2000, help="Optimization iterations per case.")
    parser.add_argument("--rebase-every", type=int, default=100, help="NVP rebase interval.")
    parser.add_argument("--save-every", type=int, default=200, help="Metric snapshot interval passed to training.")
    parser.add_argument("--checkpoint-every", type=int, default=0, help="Checkpoint interval. Use <=0 to disable.")
    parser.add_argument("--binary-thresh", type=float, default=0.5, help="Threshold for hard Dice.")
    parser.add_argument("--global-translation-lr", type=float, default=0.01)
    parser.add_argument("--global-rotation-lr", type=float, default=0.005)
    parser.add_argument("--global-scale-lr", type=float, default=0.005)
    parser.add_argument("--global-scale-min", type=float, default=0.5)
    parser.add_argument("--global-scale-max", type=float, default=2.0)
    parser.add_argument("--no-optimize-global-transform", action="store_true")
    parser.add_argument("--flip-mesh-xy", action="store_true", help="Pass --flip-mesh-xy to the clean trainer. Default is off for this GHD dataset.")
    parser.add_argument("--alpha", type=float, default=None, help="Optional soft occupancy alpha override.")
    parser.add_argument("--halfspace-bias", type=float, default=None, help="Optional half-space bias override.")
    parser.add_argument(
        "--soft-boundary-width",
        type=float,
        default=0.0,
        help=(
            "Explicit AABB expansion width in world units. The batch default is 0.0 to match the stable "
            "single-case OccTetHeart command and avoid the training script's auto-expanded boundary width."
        ),
    )
    parser.add_argument("--tetra-chunk-size", type=int, default=256, help="Tetrahedra processed per renderer block chunk.")
    parser.add_argument(
        "--block-size",
        type=int,
        nargs=3,
        default=(12, 12, 12),
        metavar=("Z", "Y", "X"),
        help="3D renderer block size in Z Y X order.",
    )
    parser.add_argument(
        "--render-mode",
        choices=("prob_union", "sum", "sum_clip", "max"),
        default=None,
        help="Optional soft occupancy aggregation override.",
    )
    parser.add_argument("--patient", action="append", default=None, help="Patient id to include. Can be repeated.")
    parser.add_argument("--case", action="append", default=None, help="Case prefix to include. Can be repeated.")
    parser.add_argument("--max-cases", type=int, default=None, help="Limit number of cases for testing.")
    parser.add_argument("--overwrite", action="store_true", help="Re-run cases even if their case summary already exists.")
    parser.add_argument("--fail-fast", action="store_true", help="Stop at the first failed case.")
    parser.add_argument("--dry-run", action="store_true", help="Only write the planned command list; do not train.")
    return parser.parse_args()


def strip_label_suffix(path: Path) -> str:
    name = path.name
    if not name.endswith(LABEL_SUFFIX):
        raise ValueError(f"Unexpected label filename: {path}")
    return name[: -len(LABEL_SUFFIX)]


def discover_cases(data_root: Path, patients_filter=None, cases_filter=None, max_cases=None):
    patients_filter = set(patients_filter or [])
    cases_filter = set(cases_filter or [])
    discovered = []

    for patient_dir in sorted(p for p in data_root.iterdir() if p.is_dir()):
        patient_id = patient_dir.name
        if patients_filter and patient_id not in patients_filter:
            continue

        labels = sorted(patient_dir.glob(f"*{LABEL_SUFFIX}"))
        for label_path in labels:
            case_id = strip_label_suffix(label_path)
            if cases_filter and case_id not in cases_filter:
                continue

            mesh_path = patient_dir / f"{case_id}{MESH_SUFFIX}"
            discovered.append(
                {
                    "patient_id": patient_id,
                    "case_id": case_id,
                    "phase": infer_phase(case_id),
                    "label_nifti": label_path,
                    "mesh": mesh_path,
                    "mesh_found": mesh_path.exists(),
                }
            )

            if max_cases is not None and len(discovered) >= int(max_cases):
                return discovered

    return discovered


def infer_phase(case_id: str):
    lowered = case_id.lower()
    if "_ed_" in lowered or lowered.endswith("_ed"):
        return "ED"
    if "_es_" in lowered or lowered.endswith("_es"):
        return "ES"
    return None


def latest_training_summary(case_output_dir: Path):
    summaries = list(case_output_dir.glob("*/training_summary.json"))
    if not summaries:
        summaries = list(case_output_dir.rglob("training_summary.json"))
    if not summaries:
        return None
    return max(summaries, key=lambda path: path.stat().st_mtime)


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def optional_float(value):
    if value is None:
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(value) or math.isinf(value):
        return None
    return value


def extract_dice(training_summary):
    initial_full = optional_float(training_summary.get("initial_full_volume_hard_dice_score"))
    initial_roi = optional_float(training_summary.get("initial_hard_dice_score_roi"))
    final_roi = optional_float(training_summary.get("final_hard_dice_score_roi"))
    final_full = optional_float(training_summary.get("hard_dice_score"))

    initial = initial_full
    if initial is None:
        initial = initial_roi

    final = final_full
    if final is None:
        final = final_roi

    improvement = None
    if initial is not None and final is not None:
        improvement = final - initial

    return {
        "initial_hard_dice": initial,
        "final_hard_dice": final,
        "hard_dice_improvement": improvement,
        "initial_roi_hard_dice": initial_roi,
        "final_roi_hard_dice": final_roi,
        "initial_full_volume_hard_dice": initial_full,
        "final_full_volume_hard_dice": final_full,
        "final_soft_dice": optional_float(training_summary.get("final_soft_dice_score")),
    }


def extract_mesh_quality(training_summary):
    final_eval = training_summary.get("final_evaluation")
    if not isinstance(final_eval, dict):
        final_eval = {}

    neg_vol = training_summary.get("final_flipped_tetrahedra_count")
    if neg_vol is None:
        neg_vol = final_eval.get("negative_signed_volume_count")

    return {
        "mesh_quality_mr_p05": optional_float(final_eval.get("mean_ratio_p05")),
        "mesh_quality_mr_p50": optional_float(final_eval.get("mean_ratio_p50")),
        "mesh_quality_rr_p05": optional_float(final_eval.get("radius_ratio_p05")),
        "mesh_quality_rr_p50": optional_float(final_eval.get("radius_ratio_p50")),
        "mesh_quality_rr_lt_0p2_count": optional_float(final_eval.get("radius_ratio_lt_0p2_count")),
        "mesh_quality_neg_vol_count": optional_float(neg_vol),
        "final_evaluation_path": training_summary.get("final_evaluation_path"),
    }


def numeric_stats(values, cases, key):
    pairs = [(case, optional_float(case.get(key))) for case in cases]
    valid = [(case, value) for case, value in pairs if value is not None]
    if not valid:
        return {
            "count": 0,
            "mean": None,
            "std": None,
            "median": None,
            "min": None,
            "max": None,
            "min_case": None,
            "max_case": None,
        }

    nums = [value for _, value in valid]
    min_case, min_value = min(valid, key=lambda item: item[1])
    max_case, max_value = max(valid, key=lambda item: item[1])
    return {
        "count": len(nums),
        "mean": float(statistics.mean(nums)),
        "std": float(statistics.pstdev(nums)) if len(nums) > 1 else 0.0,
        "median": float(statistics.median(nums)),
        "min": float(min_value),
        "max": float(max_value),
        "min_case": min_case.get("case_id"),
        "max_case": max_case.get("case_id"),
    }


def numeric_total(cases, key):
    values = [optional_float(case.get(key)) for case in cases]
    values = [value for value in values if value is not None]
    if not values:
        return None
    return float(sum(values))


def build_command(args, train_script: Path, case, case_output_dir: Path):
    cmd = [
        str(args.python),
        str(train_script),
        "--mesh",
        str(case["mesh"]),
        "--label-nifti",
        str(case["label_nifti"]),
        "--label-value",
        str(args.label_value),
        "--model-path",
        str(case_output_dir),
        "--iterations",
        str(args.iterations),
        "--rebase-every",
        str(args.rebase_every),
        "--save-every",
        str(args.save_every),
        "--checkpoint-every",
        str(args.checkpoint_every),
        "--binary-thresh",
        str(args.binary_thresh),
        "--soft-boundary-width",
        str(args.soft_boundary_width),
        "--tetra-chunk-size",
        str(args.tetra_chunk_size),
        "--block-size",
        str(args.block_size[0]),
        str(args.block_size[1]),
        str(args.block_size[2]),
        "--save-final-volume",
        "--mesh-is-local",
        "--global-translation-lr",
        str(args.global_translation_lr),
        "--global-rotation-lr",
        str(args.global_rotation_lr),
        "--global-scale-lr",
        str(args.global_scale_lr),
        "--global-scale-min",
        str(args.global_scale_min),
        "--global-scale-max",
        str(args.global_scale_max),
    ]

    if args.no_optimize_global_transform:
        cmd.append("--no-optimize-global-transform")
    if args.flip_mesh_xy:
        cmd.append("--flip-mesh-xy")

    if args.alpha is not None:
        cmd.extend(["--alpha", str(args.alpha)])
    if args.halfspace_bias is not None:
        cmd.extend(["--halfspace-bias", str(args.halfspace_bias)])
    if args.render_mode is not None:
        cmd.extend(["--render-mode", str(args.render_mode)])

    return cmd


def summarize_batch(output_root: Path, args, cases):
    completed = [case for case in cases if case.get("status") in {"completed", "existing"}]
    failed = [case for case in cases if case.get("status") == "failed"]
    skipped = [case for case in cases if str(case.get("status", "")).startswith("skipped")]
    dry_run = [case for case in cases if case.get("status") == "dry_run"]

    aggregate = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "data_root": str(Path(args.data_root).resolve()),
        "output_root": str(output_root.resolve()),
        "train_script": str(Path(args.train_script).resolve()),
        "label_value": int(args.label_value),
        "iterations": int(args.iterations),
        "rebase_every": int(args.rebase_every),
        "binary_thresh": float(args.binary_thresh),
        "global_translation_lr": float(args.global_translation_lr),
        "global_rotation_lr": float(args.global_rotation_lr),
        "global_scale_lr": float(args.global_scale_lr),
        "global_scale_min": float(args.global_scale_min),
        "global_scale_max": float(args.global_scale_max),
        "optimize_global_transform": not bool(args.no_optimize_global_transform),
        "flip_mesh_xy": bool(args.flip_mesh_xy),
        "soft_boundary_width": float(args.soft_boundary_width),
        "tetra_chunk_size": int(args.tetra_chunk_size),
        "block_size_zyx": [int(v) for v in args.block_size],
        "num_cases_discovered": len(cases),
        "num_completed": len(completed),
        "num_failed": len(failed),
        "num_skipped": len(skipped),
        "num_dry_run": len(dry_run),
        "stats": {
            "initial_hard_dice": numeric_stats([], completed, "initial_hard_dice"),
            "final_hard_dice": numeric_stats([], completed, "final_hard_dice"),
            "hard_dice_improvement": numeric_stats([], completed, "hard_dice_improvement"),
            "final_soft_dice": numeric_stats([], completed, "final_soft_dice"),
            "mesh_quality_mr_p05": numeric_stats([], completed, "mesh_quality_mr_p05"),
            "mesh_quality_mr_p50": numeric_stats([], completed, "mesh_quality_mr_p50"),
            "mesh_quality_rr_p05": numeric_stats([], completed, "mesh_quality_rr_p05"),
            "mesh_quality_rr_p50": numeric_stats([], completed, "mesh_quality_rr_p50"),
            "mesh_quality_rr_lt_0p2_count": numeric_stats([], completed, "mesh_quality_rr_lt_0p2_count"),
            "mesh_quality_neg_vol_count": numeric_stats([], completed, "mesh_quality_neg_vol_count"),
        },
        "mesh_quality_totals": {
            "rr_lt_0p2_count": numeric_total(completed, "mesh_quality_rr_lt_0p2_count"),
            "negative_signed_volume_count": numeric_total(completed, "mesh_quality_neg_vol_count"),
        },
        "cases": cases,
    }
    write_json(output_root / AGGREGATE_SUMMARY_NAME, aggregate)
    return aggregate


def main():
    args = parse_args()
    repo_root = Path(__file__).resolve().parent
    data_root = Path(args.data_root).expanduser().resolve()
    if not data_root.exists():
        raise FileNotFoundError(f"--data-root does not exist: {data_root}")

    output_root = Path(args.output_root) if args.output_root else repo_root / "output" / f"acdc_testing_ghd_global_clean_batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_root = output_root.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    train_script = Path(args.train_script)
    if not train_script.is_absolute():
        train_script = repo_root / train_script
    train_script = train_script.resolve()
    if not train_script.exists():
        raise FileNotFoundError(f"--train-script does not exist: {train_script}")

    discovered = discover_cases(data_root, args.patient, args.case, args.max_cases)
    if not discovered:
        raise RuntimeError(f"No cases found under {data_root}")

    case_results = []
    for index, case in enumerate(discovered, start=1):
        case_output_dir = output_root / case["patient_id"] / case["case_id"]
        case_summary_path = case_output_dir / CASE_SUMMARY_NAME
        command = build_command(args, train_script, case, case_output_dir)

        result = {
            "index": index,
            "patient_id": case["patient_id"],
            "case_id": case["case_id"],
            "phase": case["phase"],
            "mesh": str(case["mesh"].resolve()),
            "label_nifti": str(case["label_nifti"].resolve()),
            "output_dir": str(case_output_dir),
            "case_summary_path": str(case_summary_path),
            "command": command,
        }

        if not case["mesh_found"]:
            result["status"] = "skipped_missing_mesh"
            result["error"] = f"Missing mesh: {case['mesh']}"
            case_results.append(result)
            summarize_batch(output_root, args, case_results)
            continue

        if case_summary_path.exists() and not args.overwrite:
            existing = load_json(case_summary_path)
            existing_summary_path = existing.get("training_summary_path")
            if existing_summary_path and Path(existing_summary_path).exists():
                existing_training_summary = load_json(Path(existing_summary_path))
                existing.update(extract_mesh_quality(existing_training_summary))
                existing["final_flipped_tetrahedra_count"] = existing.get("mesh_quality_neg_vol_count")
                write_json(case_summary_path, existing)
            existing["status"] = "existing"
            case_results.append(existing)
            summarize_batch(output_root, args, case_results)
            print(f"[{index}/{len(discovered)}] existing {case['case_id']}")
            continue

        if args.dry_run:
            result["status"] = "dry_run"
            case_results.append(result)
            summarize_batch(output_root, args, case_results)
            print(f"[{index}/{len(discovered)}] dry-run {case['case_id']}")
            continue

        case_output_dir.mkdir(parents=True, exist_ok=True)
        start_time = time.time()
        print(f"[{index}/{len(discovered)}] training {case['case_id']}", flush=True)
        completed = subprocess.run(
            command,
            cwd=str(repo_root),
            text=True,
        )

        duration_sec = time.time() - start_time
        result["duration_sec"] = float(duration_sec)
        result["returncode"] = int(completed.returncode)

        training_summary_path = latest_training_summary(case_output_dir)
        if completed.returncode != 0:
            result["status"] = "failed"
            result["error"] = f"Training exited with return code {completed.returncode}"
            if training_summary_path is not None:
                result["training_summary_path"] = str(training_summary_path)
            write_json(case_summary_path, result)
            case_results.append(result)
            summarize_batch(output_root, args, case_results)
            if args.fail_fast:
                raise RuntimeError(result["error"])
            continue

        if training_summary_path is None:
            result["status"] = "failed"
            result["error"] = "Training completed but no training_summary.json was found."
            write_json(case_summary_path, result)
            case_results.append(result)
            summarize_batch(output_root, args, case_results)
            if args.fail_fast:
                raise RuntimeError(result["error"])
            continue

        training_summary = load_json(training_summary_path)
        result.update(extract_dice(training_summary))
        result.update(extract_mesh_quality(training_summary))
        result["status"] = "completed"
        result["training_summary_path"] = str(training_summary_path)
        result["run_dir"] = str(training_summary_path.parent)
        result["tetra_final_path"] = training_summary.get("tetra_final_path")
        result["pred_mask_final_nifti"] = training_summary.get("pred_mask_final_nifti")
        result["gt_mask_nifti"] = training_summary.get("gt_mask_nifti")
        result["rebase_count"] = training_summary.get("rebase_count")
        result["final_flipped_tetrahedra_count"] = result.get("mesh_quality_neg_vol_count")

        write_json(case_summary_path, result)
        case_results.append(result)
        summarize_batch(output_root, args, case_results)
        print(
            f"[{index}/{len(discovered)}] done {case['case_id']} "
            f"init={result.get('initial_hard_dice')} "
            f"final={result.get('final_hard_dice')} "
            f"improve={result.get('hard_dice_improvement')} "
            f"flipped={result.get('final_flipped_tetrahedra_count')}"
        )

    aggregate = summarize_batch(output_root, args, case_results)
    print(f"[ok] aggregate_summary={output_root / AGGREGATE_SUMMARY_NAME}")
    print(f"[ok] completed={aggregate['num_completed']} failed={aggregate['num_failed']} skipped={aggregate['num_skipped']}")


if __name__ == "__main__":
    main()
