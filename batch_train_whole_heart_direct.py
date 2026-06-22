#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shlex
import subprocess
from pathlib import Path

import numpy as np

from eval_tet_mesh import evaluate_tetra_mesh_quality
from train_whole_heart_direct import _build_train_cmd, _center_mesh_like_training_grid, _script_dir


def _default_data_root() -> Path:
    local = _script_dir() / "whole_heart"
    if local.exists():
        return local
    return _script_dir().parent / "whole_heart"


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


def _discover_cases(data_root: Path, selected: set[str]) -> list[dict]:
    cases: list[dict] = []
    for case_dir in sorted(path for path in data_root.iterdir() if path.is_dir()):
        case_id = case_dir.name
        if selected and case_id not in selected:
            continue
        labels = sorted(path for path in case_dir.iterdir() if _is_nifti(path))
        meshes = sorted(case_dir.glob("*.msh"))
        if len(labels) != 1 or len(meshes) != 1:
            cases.append(
                {
                    "case_id": case_id,
                    "case_dir": case_dir,
                    "status": "invalid_inputs",
                    "error": f"Expected exactly one NIfTI and one .msh, found {len(labels)} NIfTI and {len(meshes)} .msh.",
                }
            )
            continue
        cases.append(
            {
                "case_id": case_id,
                "case_dir": case_dir,
                "label_nifti": labels[0],
                "mesh": meshes[0],
                "status": "ready",
            }
        )
    if not cases:
        raise RuntimeError(f"No whole-heart case directories found under {data_root}.")
    return cases


def _case_completed(train_dir: Path) -> bool:
    return (train_dir / "tetra_final_soft_volume_rebase.msh").exists() and (train_dir / "training_summary.json").exists()


def _optional_float(value):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(value):
        return None
    return value


def _load_training_summary(train_dir: Path) -> dict:
    summary_path = train_dir / "training_summary.json"
    if not summary_path.exists():
        return {}
    return json.loads(summary_path.read_text(encoding="utf-8"))


def _mesh_quality_record(train_dir: Path) -> dict:
    final_mesh = train_dir / "tetra_final_soft_volume_rebase.msh"
    summary = _load_training_summary(train_dir)
    final_eval = summary.get("final_evaluation")
    eval_path = summary.get("final_evaluation_path")
    if not isinstance(final_eval, dict):
        eval_path = train_dir / "final_evaluation.json"
        try:
            final_eval = evaluate_tetra_mesh_quality(final_mesh, json_out=eval_path)
        except Exception as exc:
            return {
                "final_mesh": str(final_mesh.resolve()) if final_mesh.exists() else None,
                "mesh_quality_error": str(exc),
                "final_evaluation_path": str(eval_path.resolve()),
            }

    neg_vol = summary.get("final_flipped_tetrahedra_count")
    if neg_vol is None:
        neg_vol = final_eval.get("negative_signed_volume_count")

    return {
        "final_mesh": str(final_mesh.resolve()) if final_mesh.exists() else None,
        "training_summary": str((train_dir / "training_summary.json").resolve()) if (train_dir / "training_summary.json").exists() else None,
        "mesh_quality_mr_p05": _optional_float(final_eval.get("mean_ratio_p05")),
        "mesh_quality_mr_p50": _optional_float(final_eval.get("mean_ratio_p50")),
        "mesh_quality_rr_p05": _optional_float(final_eval.get("radius_ratio_p05")),
        "mesh_quality_rr_p50": _optional_float(final_eval.get("radius_ratio_p50")),
        "mesh_quality_rr_lt_0p2_count": _optional_float(final_eval.get("radius_ratio_lt_0p2_count")),
        "mesh_quality_neg_vol_count": _optional_float(neg_vol),
        "final_flipped_tetrahedra_count": _optional_float(neg_vol),
        "final_evaluation_path": None if eval_path is None else str(Path(eval_path).resolve()),
    }


def _result_values(results: list[dict], key: str) -> list[float]:
    return [
        value
        for value in (_optional_float(item.get(key)) for item in results)
        if value is not None
    ]


def _result_stats(results: list[dict], key: str) -> dict:
    values = _result_values(results, key)
    if not values:
        return {"count": 0, "mean": None, "min": None, "max": None}
    return {
        "count": len(values),
        "mean": float(np.mean(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
    }


def _result_total(results: list[dict], key: str):
    values = _result_values(results, key)
    if not values:
        return None
    return float(np.sum(values))


def _write_batch_summary(output_root: Path, data_root: Path, cases: list[dict], failures: int, results: list[dict]) -> None:
    summary = {
        "data_root": str(data_root),
        "output_root": str(output_root),
        "num_cases": len(cases),
        "num_failed": int(failures),
        "mesh_quality_stats": {
            "mr_p05": _result_stats(results, "mesh_quality_mr_p05"),
            "mr_p50": _result_stats(results, "mesh_quality_mr_p50"),
            "rr_p05": _result_stats(results, "mesh_quality_rr_p05"),
            "rr_p50": _result_stats(results, "mesh_quality_rr_p50"),
            "rr_lt_0p2_count": _result_stats(results, "mesh_quality_rr_lt_0p2_count"),
            "neg_vol_count": _result_stats(results, "mesh_quality_neg_vol_count"),
        },
        "mesh_quality_totals": {
            "rr_lt_0p2_count": _result_total(results, "mesh_quality_rr_lt_0p2_count"),
            "negative_signed_volume_count": _result_total(results, "mesh_quality_neg_vol_count"),
        },
        "results": results,
    }
    (output_root / "whole_heart_direct_batch_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Batch direct MRI-TET training for whole_heart/pat*/ cases. Each case must contain one .nii/.nii.gz "
            "label and one corresponding .msh mesh. Meshes are centered like train_whole_heart_direct.py; no registration is used."
        )
    )
    parser.add_argument("--data-root", type=Path, default=_default_data_root())
    parser.add_argument("--output-root", type=Path, default=_script_dir() / "outputs" / "whole_heart_direct_batch")
    parser.add_argument("--case", action="append", default=[], help="Case id such as pat0. Repeatable.")
    parser.add_argument("--coordinate-scale", type=float, default=0.01)
    parser.add_argument("--overwrite-centered-mesh", action="store_true")

    parser.add_argument("--label-value", type=int, default=-1)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--save-every", type=int, default=200)
    parser.add_argument("--checkpoint-every", type=int, default=0)
    parser.add_argument(
        "--rebase-every",
        type=int,
        default=0,
        help="Set <=0 to disable rebase by passing iterations+1 to the trainer. Default disables rebase.",
    )
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
    parser.add_argument("--no-timestamp-output", action="store_true", default=True)
    parser.add_argument("--extra-args", default="")

    parser.add_argument("--overwrite-training", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--stop-on-error", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    data_root = args.data_root.resolve()
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    cases = _discover_cases(data_root, set(args.case))
    print(f"[whole_heart_batch] data_root={data_root}", flush=True)
    print(f"[whole_heart_batch] output_root={output_root}", flush=True)
    print(f"[whole_heart_batch] num_cases={len(cases)} iterations={args.iterations} rebase=disabled", flush=True)

    results: list[dict] = []
    failures = 0
    for index, case in enumerate(cases, start=1):
        case_id = case["case_id"]
        case_root = output_root / case_id
        preprocessed_root = case_root / "preprocessed"
        train_dir = case_root / "train"
        print(f"[case] {index}/{len(cases)} {case_id}", flush=True)

        if case.get("status") != "ready":
            failures += 1
            record = {"case_id": case_id, "status": "failed", "error": case["error"]}
            results.append(record)
            print(f"[failed] {case_id}: {case['error']}", flush=True)
            if args.stop_on_error:
                break
            continue

        label_nifti = Path(case["label_nifti"]).resolve()
        mesh_path = Path(case["mesh"]).resolve()
        centered_mesh = preprocessed_root / f"{mesh_path.stem}_centered_scale{args.coordinate_scale:g}.msh"
        manifest_path = preprocessed_root / "whole_heart_centering_manifest.json"

        try:
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

            run_args = argparse.Namespace(**vars(args))
            run_args.rebase_every = int(args.rebase_every) if int(args.rebase_every) > 0 else int(args.iterations) + 1
            cmd = _build_train_cmd(run_args, centered_mesh, label_nifti, train_dir)
            print(f"[paths] label={label_nifti} mesh={mesh_path}", flush=True)
            print(f"[centered] mesh={centered_mesh}", flush=True)
            print(f"[cmd] {' '.join(shlex.quote(part) for part in cmd)}", flush=True)

            if args.dry_run:
                status = "dry_run"
            elif _case_completed(train_dir) and not args.overwrite_training:
                status = "skipped_completed"
                print(f"[train] skip completed {case_id}", flush=True)
            else:
                completed = subprocess.run(cmd, cwd=_script_dir(), check=False)
                if completed.returncode != 0:
                    raise RuntimeError(f"training failed with exit code {completed.returncode}")
                status = "completed"

            record = {
                "case_id": case_id,
                "status": status,
                "label_nifti": str(label_nifti),
                "input_mesh": str(mesh_path),
                "centered_mesh": str(centered_mesh.resolve()),
                "train_dir": str(train_dir.resolve()),
                "coordinate_scale": float(args.coordinate_scale),
                "rebase_every_passed": int(run_args.rebase_every),
                "centered_manifest": manifest,
            }
            if status != "dry_run" and _case_completed(train_dir):
                record.update(_mesh_quality_record(train_dir))
            results.append(record)
        except Exception as exc:
            failures += 1
            record = {"case_id": case_id, "status": "failed", "error": str(exc)}
            results.append(record)
            print(f"[failed] {case_id}: {exc}", flush=True)
            if args.stop_on_error:
                break
        finally:
            _write_batch_summary(output_root, data_root, cases, failures, results)

    print(f"[whole_heart_batch] summary={output_root / 'whole_heart_direct_batch_summary.json'}", flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
