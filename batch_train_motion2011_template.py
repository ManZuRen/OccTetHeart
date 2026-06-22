#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import shlex
import subprocess
import sys
from pathlib import Path

import meshio
import numpy as np
from scipy.spatial import cKDTree

from eval_tet_mesh import evaluate_tetra_mesh_quality
from utils.template_registration import align_template_mesh_to_label, label_boundary_direction_points, load_template_registration_inputs


def _script_dir() -> Path:
    return Path(__file__).resolve().parent


def _default_dataset_root() -> Path:
    local = _script_dir() / "motion_data_seg_rvmyo_zflip"
    if local.exists():
        return local
    return _script_dir().parent / "motion2011" / "motion_data_seg_rvmyo_zflip"


def _default_template_msh() -> Path:
    local = _script_dir() / "biv_template_generation" / "heart_sur_medium_uniform_tet32k_labeled.msh"
    if local.exists():
        return local
    return _script_dir().parent / "biv_template_generation" / "heart_sur_medium_uniform_tet32k_labeled.msh"


def _is_nifti(path: Path) -> bool:
    name = path.name.lower()
    return path.is_file() and (name.endswith(".nii") or name.endswith(".nii.gz"))


def _safe_id(path: Path) -> str:
    name = path.name
    if name.endswith(".nii.gz"):
        name = name[:-7]
    else:
        name = path.stem
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name)


def _case_id_for_label(label_path: Path, data_root: Path) -> str:
    rel = label_path.relative_to(data_root)
    parts = list(rel.parts)
    if len(parts) >= 3 and parts[1].lower() == "csax":
        return f"{_safe_id(Path(parts[0]))}__cSAX__{_safe_id(label_path)}"
    return "__".join(_safe_id(Path(part)) for part in parts)


def _time_index(label_path: Path) -> int:
    match = re.search(r"(\d+)", label_path.name)
    return int(match.group(1)) if match else 0


def _discover_groups(data_root: Path, label_glob: str, selected: set[str]) -> dict[str, list[Path]]:
    labels = sorted(path for path in data_root.glob(label_glob) if _is_nifti(path))
    groups: dict[str, list[Path]] = {}
    for label in labels:
        rel = label.relative_to(data_root)
        subject = rel.parts[0] if rel.parts else label.parent.name
        if selected and subject not in selected:
            continue
        groups.setdefault(subject, []).append(label)
    for subject in list(groups):
        groups[subject].sort(key=lambda p: (_time_index(p), str(p)))
    if not groups:
        raise RuntimeError(f"No motion2011 labels found under {data_root} with --label-glob {label_glob!r}.")
    return dict(sorted(groups.items()))


def _random_subsample(points: np.ndarray, count: int, rng: np.random.Generator) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    if points.shape[0] <= int(count):
        return points
    idx = rng.choice(points.shape[0], size=int(count), replace=False)
    return points[idx]


def _solve_similarity_from_pairs(
    source: np.ndarray,
    target: np.ndarray,
    scale_min: float,
    scale_max: float,
) -> tuple[np.ndarray, float, np.ndarray]:
    src_center = source.mean(axis=0)
    tgt_center = target.mean(axis=0)
    src0 = source - src_center[None, :]
    tgt0 = target - tgt_center[None, :]
    cov = src0.T @ tgt0 / max(source.shape[0], 1)
    u, singular_values, vt = np.linalg.svd(cov)
    reflect = np.eye(3, dtype=np.float64)
    if np.linalg.det(vt.T @ u.T) < 0:
        reflect[-1, -1] = -1.0
    rot = vt.T @ reflect @ u.T
    denom = float(np.sum(src0 * src0) / max(source.shape[0], 1))
    scale = float(np.sum(singular_values * np.diag(reflect)) / max(denom, 1e-12))
    scale = float(np.clip(scale, scale_min, scale_max))
    trans = tgt_center - scale * (src_center @ rot.T)
    return rot.astype(np.float64), scale, trans.astype(np.float64)


def _align_followup_from_previous_mesh(
    source_msh: Path,
    label_path: Path,
    out_msh: Path,
    args: argparse.Namespace,
) -> dict:
    rng = np.random.default_rng(int(args.seed))
    source_points = int(args.source_points_followup)
    source_mesh, source_nodes, _source_faces, source_samples = load_template_registration_inputs(
        source_msh.resolve(),
        source_points=source_points,
        seed=int(args.seed),
    )
    dst = label_boundary_direction_points(label_path, args.registration_labels, args.target_rescale)
    dst = _random_subsample(dst, int(args.target_points), rng)

    rot = np.eye(3, dtype=np.float64)
    scale = 1.0
    trans = np.zeros(3, dtype=np.float64)
    scale_min = float(args.follow_scale_min)
    scale_max = float(args.follow_scale_max)
    tree = cKDTree(dst)
    last_mean = None
    iterations = 0
    for iterations in range(1, int(args.follow_registration_maxiter) + 1):
        moved = (source_samples @ rot.T) * scale + trans[None, :]
        distances, nn_idx = tree.query(moved, k=1)
        keep = distances <= np.quantile(distances, float(args.registration_trim))
        if int(np.count_nonzero(keep)) < 50:
            keep = np.ones_like(distances, dtype=bool)

        delta_rot, delta_scale, delta_trans = _solve_similarity_from_pairs(
            moved[keep],
            dst[nn_idx[keep]],
            scale_min=scale_min,
            scale_max=scale_max,
        )
        old_scale = scale
        new_scale = float(np.clip(delta_scale * scale, scale_min, scale_max))
        effective_delta_scale = new_scale / max(old_scale, 1e-12)
        rot = delta_rot @ rot
        scale = new_scale
        trans = (trans @ delta_rot.T) * effective_delta_scale + delta_trans

        mean_distance = float(distances[keep].mean())
        if last_mean is not None and abs(last_mean - mean_distance) < float(args.registration_tol):
            break
        last_mean = mean_distance

    aligned_nodes = (source_nodes @ rot.T) * scale + trans[None, :]
    out_msh.parent.mkdir(parents=True, exist_ok=True)
    meshio.write(
        str(out_msh),
        meshio.Mesh(
            points=aligned_nodes,
            cells=source_mesh.cells,
            point_data=source_mesh.point_data,
            cell_data=source_mesh.cell_data,
            field_data=source_mesh.field_data,
        ),
        file_format="gmsh22",
        binary=True,
    )
    moved_src = (source_samples @ rot.T) * scale + trans[None, :]
    final_distances, _ = tree.query(moved_src, k=1)
    record = {
        "label_nifti": str(label_path.resolve()),
        "aligned_mesh": str(out_msh.resolve()),
        "registration_labels": [float(v) for v in args.registration_labels],
        "target_rescale": float(args.target_rescale),
        "iterations": int(iterations),
        "icp_scale": float(scale),
        "translation": trans.astype(float).tolist(),
        "rotation": rot.astype(float).tolist(),
        "mean_nn": float(final_distances.mean()),
        "p90_nn": float(np.quantile(final_distances, 0.9)),
        "bbox_min": aligned_nodes.min(axis=0).astype(float).tolist(),
        "bbox_max": aligned_nodes.max(axis=0).astype(float).tolist(),
        "source_msh": str(source_msh.resolve()),
        "source_unit_scale": 1.0,
        "followup": True,
        "scale_min": scale_min,
        "scale_max": scale_max,
        "registration": "motion2011_followup_identity_init_similarity_icp",
    }
    out_msh.with_suffix(".registration.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    return record


def _align_mesh_to_label(
    source_msh: Path,
    label_path: Path,
    out_msh: Path,
    args: argparse.Namespace,
    followup: bool,
) -> dict:
    if followup:
        return _align_followup_from_previous_mesh(
            source_msh=source_msh,
            label_path=label_path,
            out_msh=out_msh,
            args=args,
        )

    source_points = int(args.source_points_followup if followup else args.source_points)
    template_mesh, template_nodes, template_faces, template_samples = load_template_registration_inputs(
        source_msh.resolve(),
        source_points=source_points,
        seed=int(args.seed),
    )
    source_unit_scale = 1.0 if followup else float(args.target_rescale)
    if source_unit_scale != 1.0:
        template_mesh.points = np.asarray(template_mesh.points, dtype=np.float64).copy()
        template_mesh.points[:, :3] *= source_unit_scale
        template_nodes = template_nodes * source_unit_scale
        template_samples = template_samples * source_unit_scale
    record = align_template_mesh_to_label(
        template_mesh=template_mesh,
        template_nodes=template_nodes,
        template_boundary_faces=template_faces,
        template_surface_samples=template_samples,
        label_nii=label_path,
        out_msh=out_msh,
        out_transform_json=out_msh.with_suffix(".registration.json"),
        registration_labels=args.registration_labels,
        target_rescale=args.target_rescale,
        target_points=args.target_points,
        coarse_points=args.coarse_points,
        maxiter=int(args.follow_registration_maxiter if followup else args.registration_maxiter),
        tol=float(args.registration_tol),
        trim=float(args.registration_trim),
        scale_min=float(args.follow_scale_min if followup else args.scale_min),
        scale_max=float(args.follow_scale_max if followup else args.scale_max),
        seed=int(args.seed),
    )
    record["source_msh"] = str(source_msh.resolve())
    record["source_unit_scale"] = float(source_unit_scale)
    record["followup"] = bool(followup)
    record["scale_min"] = float(args.follow_scale_min if followup else args.scale_min)
    record["scale_max"] = float(args.follow_scale_max if followup else args.scale_max)
    record["registration"] = "current_template_direction_space_similarity_icp_source_scaled"
    out_msh.with_suffix(".registration.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    return record


def _latest_final_mesh(training_dir: Path) -> Path:
    candidates = list(training_dir.glob("tetra_final_soft_volume_rebase.msh"))
    candidates.extend(training_dir.glob("*/tetra_final_soft_volume_rebase.msh"))
    if not candidates:
        raise FileNotFoundError(f"Missing trained final mesh under: {training_dir}")
    return max(candidates, key=lambda path: path.stat().st_mtime)


def _latest_training_summary_path(training_dir: Path) -> Path | None:
    candidates = list(training_dir.glob("training_summary.json"))
    candidates.extend(training_dir.glob("*/training_summary.json"))
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime)


def _has_completed_training(training_dir: Path) -> bool:
    try:
        _latest_final_mesh(training_dir)
    except FileNotFoundError:
        return False
    return True


def _registration_labels_match(actual: object, expected: list[float]) -> bool:
    if not isinstance(actual, list) or len(actual) != len(expected):
        return False
    return all(abs(float(a) - float(b)) < 1e-8 for a, b in zip(actual, expected))


def _load_current_registration(
    aligned_msh: Path,
    source_msh: Path,
    label_path: Path,
    args: argparse.Namespace,
    followup: bool,
) -> dict | None:
    registration_json = aligned_msh.with_suffix(".registration.json")
    if not aligned_msh.exists() or not registration_json.exists():
        return None
    record = json.loads(registration_json.read_text(encoding="utf-8"))
    expected_registration = (
        "motion2011_followup_identity_init_similarity_icp"
        if followup
        else "current_template_direction_space_similarity_icp_source_scaled"
    )
    if record.get("registration") != expected_registration:
        return None
    if Path(record.get("source_msh", "")).resolve() != source_msh.resolve():
        return None
    if Path(record.get("label_nifti", "")).resolve() != label_path.resolve():
        return None
    expected_source_unit_scale = 1.0 if followup else float(args.target_rescale)
    if abs(float(record.get("source_unit_scale", -1.0)) - expected_source_unit_scale) > 1e-8:
        return None
    expected_scale_min = float(args.follow_scale_min if followup else args.scale_min)
    expected_scale_max = float(args.follow_scale_max if followup else args.scale_max)
    if abs(float(record.get("scale_min", -1.0)) - expected_scale_min) > 1e-8:
        return None
    if abs(float(record.get("scale_max", -1.0)) - expected_scale_max) > 1e-8:
        return None
    if abs(float(record.get("target_rescale", -1.0)) - float(args.target_rescale)) > 1e-8:
        return None
    if not _registration_labels_match(record.get("registration_labels"), [float(v) for v in args.registration_labels]):
        return None
    return record


def _train_cmd(
    train_script: Path,
    aligned_msh: Path,
    label_path: Path,
    model_path: Path,
    iterations: int,
    rebase_every: int,
    args: argparse.Namespace,
    passthrough: list[str],
) -> list[str]:
    cmd = [
        sys.executable,
        str(train_script),
        "--mesh",
        str(aligned_msh.resolve()),
        "--mesh-is-local",
        "--label-nifti",
        str(label_path.resolve()),
        "--model-path",
        str(model_path.resolve()),
        "--no-timestamp-output",
        "--label-value",
        str(args.label_value),
        "--sh-degree",
        str(args.sh_degree),
        "--iterations",
        str(iterations),
        "--save-every",
        str(args.save_every),
        "--checkpoint-every",
        str(args.checkpoint_every),
        "--rebase-every",
        str(rebase_every),
        "--alpha",
        str(args.alpha),
        "--halfspace-bias",
        str(args.halfspace_bias),
        "--boundary-thresh",
        str(args.boundary_thresh),
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
    if args.soft_boundary_width is not None:
        cmd.extend(["--soft-boundary-width", str(args.soft_boundary_width)])
    if args.multiply_opacity:
        cmd.append("--multiply-opacity")
    if args.global_gate_thresh is not None:
        cmd.extend(["--global-gate-thresh", str(args.global_gate_thresh)])
    if args.save_final_volume:
        cmd.append("--save-final-volume")
    if args.extra_args:
        cmd.extend(shlex.split(args.extra_args))
    cmd.extend(passthrough)
    return cmd


def _load_training_summary(train_dir: Path) -> dict:
    summary_path = _latest_training_summary_path(train_dir)
    if summary_path is None:
        return {}
    return json.loads(summary_path.read_text(encoding="utf-8"))


def _optional_float(value):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(value):
        return None
    return value


def _mesh_quality_record(summary: dict, final_mesh: Path) -> dict:
    final_eval = summary.get("final_evaluation")
    eval_path = summary.get("final_evaluation_path")
    if not isinstance(final_eval, dict):
        eval_path = final_mesh.parent / "final_evaluation.json"
        try:
            final_eval = evaluate_tetra_mesh_quality(final_mesh, json_out=eval_path)
        except Exception as exc:
            return {
                "mesh_quality_error": str(exc),
                "final_evaluation_path": str(eval_path.resolve()),
            }

    neg_vol = summary.get("final_flipped_tetrahedra_count")
    if neg_vol is None:
        neg_vol = final_eval.get("negative_signed_volume_count")

    return {
        "mesh_quality_mr_p05": _optional_float(final_eval.get("mean_ratio_p05")),
        "mesh_quality_mr_p50": _optional_float(final_eval.get("mean_ratio_p50")),
        "mesh_quality_rr_p05": _optional_float(final_eval.get("radius_ratio_p05")),
        "mesh_quality_rr_p50": _optional_float(final_eval.get("radius_ratio_p50")),
        "mesh_quality_rr_lt_0p2_count": _optional_float(final_eval.get("radius_ratio_lt_0p2_count")),
        "mesh_quality_neg_vol_count": _optional_float(neg_vol),
        "final_flipped_tetrahedra_count": _optional_float(neg_vol),
        "final_evaluation_path": None if eval_path is None else str(Path(eval_path).resolve()),
    }


def _dice_record(
    subject: str,
    time_index: int,
    label_path: Path,
    train_dir: Path,
    aligned_msh: Path,
    iterations: int,
    rebase_every: int,
    followup: bool,
) -> dict:
    summary = _load_training_summary(train_dir)
    final_mesh = _latest_final_mesh(train_dir)
    summary_path = _latest_training_summary_path(train_dir)
    record = {
        "subject": subject,
        "time_index": int(time_index),
        "time_name": label_path.name.removesuffix(".nii.gz"),
        "label_nii": str(label_path.resolve()),
        "aligned_init_msh": str(aligned_msh.resolve()),
        "final_mesh": str(final_mesh.resolve()) if final_mesh.exists() else None,
        "training_summary": str(summary_path.resolve()) if summary_path is not None else None,
        "iterations": int(iterations),
        "rebase_every": int(rebase_every),
        "followup": bool(followup),
        "initial_soft_dice_score": summary.get("initial_soft_dice_score"),
        "final_soft_dice_score": summary.get("final_soft_dice_score"),
        "initial_hard_dice_score_roi": summary.get("initial_hard_dice_score_roi"),
        "final_hard_dice_score_roi": summary.get("final_hard_dice_score_roi"),
        "hard_dice_score": summary.get("hard_dice_score"),
        "hard_dice_improvement_roi": summary.get("hard_dice_improvement_roi"),
    }
    record.update(_mesh_quality_record(summary, final_mesh))
    return record


def _record_values(records: list[dict], key: str) -> list[float]:
    return [
        value
        for value in (_optional_float(item.get(key)) for item in records)
        if value is not None
    ]


def _record_stats(records: list[dict], key: str) -> dict:
    values = _record_values(records, key)
    if not values:
        return {"count": 0, "mean": None, "min": None, "max": None}
    return {
        "count": len(values),
        "mean": float(np.mean(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
    }


def _record_total(records: list[dict], key: str):
    values = _record_values(records, key)
    if not values:
        return None
    return float(np.sum(values))


def _write_subject_dice_summary(output_root: Path, subject: str, records: list[dict]) -> None:
    dice_values = [
        item.get("final_hard_dice_score_roi")
        for item in records
        if item.get("final_hard_dice_score_roi") is not None
    ]
    soft_values = [
        item.get("final_soft_dice_score")
        for item in records
        if item.get("final_soft_dice_score") is not None
    ]
    summary = {
        "subject": subject,
        "num_timepoints": len(records),
        "mean_final_hard_dice_score_roi": float(np.mean(dice_values)) if dice_values else None,
        "min_final_hard_dice_score_roi": float(np.min(dice_values)) if dice_values else None,
        "max_final_hard_dice_score_roi": float(np.max(dice_values)) if dice_values else None,
        "mean_final_soft_dice_score": float(np.mean(soft_values)) if soft_values else None,
        "mesh_quality_stats": {
            "mr_p05": _record_stats(records, "mesh_quality_mr_p05"),
            "mr_p50": _record_stats(records, "mesh_quality_mr_p50"),
            "rr_p05": _record_stats(records, "mesh_quality_rr_p05"),
            "rr_p50": _record_stats(records, "mesh_quality_rr_p50"),
            "rr_lt_0p2_count": _record_stats(records, "mesh_quality_rr_lt_0p2_count"),
            "neg_vol_count": _record_stats(records, "mesh_quality_neg_vol_count"),
        },
        "mesh_quality_totals": {
            "rr_lt_0p2_count": _record_total(records, "mesh_quality_rr_lt_0p2_count"),
            "negative_signed_volume_count": _record_total(records, "mesh_quality_neg_vol_count"),
        },
        "timepoints": records,
    }
    subject_dir = output_root / subject
    subject_dir.mkdir(parents=True, exist_ok=True)
    (subject_dir / "dice_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        description=(
            "Sequential motion2011 training. For each subject, time_001 is aligned from the "
            "template and trained longer; each following time point is aligned from the previous "
            "trained final mesh and trained briefly."
        )
    )
    parser.add_argument("--template-msh", type=Path, default=_default_template_msh())
    parser.add_argument("--data-root", type=Path, default=_default_dataset_root())
    parser.add_argument("--label-glob", default="v*/cSAX/time_*.nii.gz")
    parser.add_argument("--output-root", type=Path, default=_script_dir() / "outputs" / "motion2011_template_batch")
    parser.add_argument("--subject", "--case", action="append", default=[], help="Subject id such as v1. Repeatable.")

    parser.add_argument("--registration-labels", nargs="+", type=float, default=[2.0])
    parser.add_argument("--target-rescale", "--coord-rescale", dest="target_rescale", type=float, default=0.01)
    parser.add_argument("--source-points", type=int, default=16000)
    parser.add_argument("--source-points-followup", type=int, default=8000)
    parser.add_argument("--target-points", type=int, default=20000)
    parser.add_argument("--coarse-points", type=int, default=5000)
    parser.add_argument("--registration-maxiter", type=int, default=80)
    parser.add_argument("--follow-registration-maxiter", type=int, default=30)
    parser.add_argument("--registration-tol", type=float, default=1e-6)
    parser.add_argument("--registration-trim", type=float, default=0.88)
    parser.add_argument("--scale-min", type=float, default=0.35)
    parser.add_argument("--scale-max", type=float, default=2.5)
    parser.add_argument("--follow-scale-min", type=float, default=0.85)
    parser.add_argument("--follow-scale-max", type=float, default=1.15)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--first-iterations", type=int, default=1000)
    parser.add_argument("--follow-iterations", type=int, default=200)
    parser.add_argument("--label-value", type=int, default=2)
    parser.add_argument("--sh-degree", type=int, default=3)
    parser.add_argument("--save-every", type=int, default=200)
    parser.add_argument("--checkpoint-every", type=int, default=0)
    parser.add_argument("--rebase-every", type=int, default=100)
    parser.add_argument(
        "--follow-rebase-every",
        type=int,
        default=0,
        help="Rebase interval for follow-up time points. Set <=0 to disable rebase for follow-up training.",
    )
    parser.add_argument("--alpha", type=float, default=32.0)
    parser.add_argument("--halfspace-bias", type=float, default=0.05)
    parser.add_argument("--boundary-thresh", type=float, default=1e-4)
    parser.add_argument("--soft-boundary-width", type=float, default=0)
    parser.add_argument("--render-mode", choices=("prob_union", "sum", "sum_clip", "max"), default="max")
    parser.add_argument("--multiply-opacity", action="store_true")
    parser.add_argument("--global-gate-thresh", type=float, default=None)
    parser.add_argument("--global-gate-steepness", type=float, default=50.0)
    parser.add_argument("--binary-thresh", type=float, default=0.5)
    parser.add_argument("--lambda-volume-dice", type=float, default=1.0)
    parser.add_argument("--lambda-volume-l1", type=float, default=0.5)
    parser.add_argument("--block-size", type=int, nargs=3, default=(12, 12, 12), metavar=("Z", "Y", "X"))
    parser.add_argument("--tetra-chunk-size", type=int, default=512)
    parser.add_argument("--save-final-volume", action="store_true", default=False)
    parser.add_argument("--extra-args", default="")

    parser.add_argument("--overwrite-registration", action="store_true")
    parser.add_argument("--overwrite-training", action="store_true")
    parser.add_argument("--registration-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--stop-on-error", action="store_true")
    return parser.parse_known_args()


def main() -> int:
    args, passthrough = parse_args()
    data_root = args.data_root.resolve()
    output_root = args.output_root.resolve()
    template_msh = args.template_msh.resolve()
    train_script = _script_dir() / "train_tet_soft_volume_rebase_nvp_global_clean.py"
    output_root.mkdir(parents=True, exist_ok=True)

    groups = _discover_groups(data_root, args.label_glob, set(args.subject))
    print(f"[motion2011] data_root={data_root}", flush=True)
    print(f"[motion2011] template_msh={template_msh}", flush=True)
    print(f"[motion2011] output_root={output_root}", flush=True)
    print(
        "[motion2011] schedule="
        f"subjects={len(groups)} first_iterations={args.first_iterations} follow_iterations={args.follow_iterations} "
        f"first_registration=current_template_direction_space_similarity_icp_source_scaled "
        f"follow_registration=motion2011_followup_identity_init_similarity_icp local_mesh=True",
        flush=True,
    )

    manifest: list[dict] = []
    failed: list[dict] = []
    for subject_index, (subject, labels) in enumerate(groups.items(), start=1):
        previous_final_mesh: Path | None = None
        subject_dice_records: list[dict] = []
        for time_index, label_path in enumerate(labels, start=1):
            case_id = _case_id_for_label(label_path, data_root)
            case_root = output_root / subject / _safe_id(label_path)
            align_dir = case_root / "alignment"
            train_dir = case_root / "train"
            aligned_msh = align_dir / f"{case_id}_aligned_init.msh"
            source_msh = template_msh if previous_final_mesh is None else previous_final_mesh
            iterations = int(args.first_iterations if previous_final_mesh is None else args.follow_iterations)
            followup = previous_final_mesh is not None
            if followup and int(args.follow_rebase_every) <= 0:
                rebase_every = iterations + 1
            else:
                rebase_every = int(args.follow_rebase_every if followup else args.rebase_every)
            if rebase_every <= 0:
                rebase_every = iterations + 1

            print(
                f"[case] subject={subject_index}/{len(groups)} {subject} "
                f"time={time_index}/{len(labels)} {label_path.name} iterations={iterations} rebase_every={rebase_every}",
                flush=True,
            )
            scale_min = float(args.follow_scale_min if followup else args.scale_min)
            scale_max = float(args.follow_scale_max if followup else args.scale_max)
            print(
                f"[paths] source_msh={source_msh.resolve()} aligned_init={aligned_msh.resolve()} train_dir={train_dir.resolve()}",
                flush=True,
            )
            print(
                f"[align-config] followup={followup} source_unit_scale={1.0 if followup else float(args.target_rescale):.6f} "
                f"scale_bounds=[{scale_min:.6f}, {scale_max:.6f}]",
                flush=True,
            )
            try:
                reg = None if args.overwrite_registration else _load_current_registration(
                    aligned_msh=aligned_msh,
                    source_msh=source_msh,
                    label_path=label_path,
                    args=args,
                    followup=followup,
                )
                if reg is None:
                    print("[align] run registration", flush=True)
                    reg = _align_mesh_to_label(
                        source_msh=source_msh,
                        label_path=label_path,
                        out_msh=aligned_msh,
                        args=args,
                        followup=followup,
                    )
                else:
                    print(f"[align] reuse registration_json={aligned_msh.with_suffix('.registration.json').resolve()}", flush=True)
                scale = reg.get("icp_scale", reg.get("scale"))
                mean_nn = reg.get("mean_nn", reg.get("icp_cost"))
                print(
                    f"[align] scale={float(scale):.6f} mean_nn={float(mean_nn):.8f} "
                    f"source={reg.get('source_msh', str(source_msh.resolve()))} aligned={aligned_msh.resolve()}",
                    flush=True,
                )

                cmd = _train_cmd(train_script, aligned_msh, label_path, train_dir, iterations, rebase_every, args, passthrough)
                dice_record = None
                if args.registration_only or args.dry_run:
                    print("[cmd] " + " ".join(shlex.quote(part) for part in cmd), flush=True)
                else:
                    has_completed_training = False if args.overwrite_training else _has_completed_training(train_dir)
                    if args.overwrite_training or not has_completed_training:
                        print(f"[train] run {case_id}", flush=True)
                        completed = subprocess.run(cmd, cwd=_script_dir(), check=False)
                        if completed.returncode != 0:
                            raise RuntimeError(f"single-case training failed with exit code {completed.returncode}")
                    else:
                        print(f"[train] skip completed {case_id}", flush=True)
                    previous_final_mesh = _latest_final_mesh(train_dir)
                    dice_record = _dice_record(
                        subject=subject,
                        time_index=time_index,
                        label_path=label_path,
                        train_dir=train_dir,
                        aligned_msh=aligned_msh,
                        iterations=iterations,
                        rebase_every=rebase_every,
                        followup=followup,
                    )
                    subject_dice_records.append(dice_record)
                    _write_subject_dice_summary(output_root, subject, subject_dice_records)

                latest_summary_path = None if args.registration_only or args.dry_run else _latest_training_summary_path(train_dir)
                record = {
                    "subject": subject,
                    "case_id": case_id,
                    "label_nii": str(label_path.resolve()),
                    "aligned_msh": str(aligned_msh.resolve()),
                    "train_dir": str(train_dir.resolve()),
                    "source_msh": str(source_msh.resolve()),
                    "iterations": iterations,
                    "rebase_every": rebase_every,
                    "followup": bool(followup),
                    "final_mesh": None if args.registration_only or args.dry_run else str(previous_final_mesh.resolve()),
                    "training_summary": str(latest_summary_path.resolve()) if latest_summary_path is not None else None,
                    "registration": reg,
                }
                if dice_record is not None:
                    record.update(
                        {
                            key: dice_record.get(key)
                            for key in (
                                "initial_soft_dice_score",
                                "final_soft_dice_score",
                                "initial_hard_dice_score_roi",
                                "final_hard_dice_score_roi",
                                "hard_dice_score",
                                "hard_dice_improvement_roi",
                                "mesh_quality_mr_p05",
                                "mesh_quality_mr_p50",
                                "mesh_quality_rr_p05",
                                "mesh_quality_rr_p50",
                                "mesh_quality_rr_lt_0p2_count",
                                "mesh_quality_neg_vol_count",
                                "final_flipped_tetrahedra_count",
                                "final_evaluation_path",
                                "mesh_quality_error",
                            )
                            if key in dice_record
                        }
                    )
                manifest.append(record)
                (output_root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
            except Exception as exc:
                failure = {"subject": subject, "case_id": case_id, "label_nii": str(label_path.resolve()), "error": str(exc)}
                failed.append(failure)
                (output_root / "failed.json").write_text(json.dumps(failed, indent=2), encoding="utf-8")
                print(f"[failed] {case_id}: {exc}", flush=True)
                if args.stop_on_error:
                    return 1
                previous_final_mesh = None

    if failed:
        print(f"[motion2011] completed with failures={len(failed)}; see {output_root / 'failed.json'}", flush=True)
        return 1
    print(f"[motion2011] completed. manifest={output_root / 'manifest.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
