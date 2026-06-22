#!/usr/bin/env python3
"""Evaluate tetrahedral mesh quality."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tet_mesh_quality import read_msh, summarize, tetra_metrics


def evaluate_tetra_mesh_quality(
    msh_path: str | Path,
    bad_radius_ratio: float = 0.2,
    json_out: str | Path | None = None,
) -> dict:
    msh_path = Path(msh_path)
    nodes, tets = read_msh(msh_path)

    metric_values = {
        "mean_ratio": [],
        "radius_ratio": [],
        "signed_volume": [],
    }
    negative = 0
    missing = 0
    for conn in tets:
        try:
            pts = [nodes[tag] for tag in conn]
        except KeyError:
            missing += 1
            continue
        metrics = tetra_metrics(pts)
        if metrics["signed_volume"] < 0.0:
            negative += 1
        for key in metric_values:
            metric_values[key].append(metrics[key])

    mr = summarize(metric_values["mean_ratio"])
    rr = summarize(metric_values["radius_ratio"])
    result = {
        "mesh": str(msh_path),
        "num_tetrahedra": int(len(tets)),
        "num_evaluated_tetrahedra": int(len(metric_values["signed_volume"])),
        "num_missing_node_tetrahedra": int(missing),
        "mean_ratio_p05": mr["p05"],
        "mean_ratio_p50": mr["median"],
        "radius_ratio_p05": rr["p05"],
        "radius_ratio_p50": rr["median"],
        "radius_ratio_lt_0p2_count": int(sum(v < bad_radius_ratio for v in metric_values["radius_ratio"])),
        "negative_signed_volume_count": int(negative),
    }
    if json_out is not None:
        Path(json_out).write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main():
    parser = argparse.ArgumentParser(description="Evaluate tetrahedral mesh quality.")
    parser.add_argument("--mesh", required=True, type=Path, help="Input tetra .msh")
    parser.add_argument("--bad-radius-ratio", type=float, default=0.2)
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()

    result = evaluate_tetra_mesh_quality(
        msh_path=args.mesh,
        bad_radius_ratio=float(args.bad_radius_ratio),
        json_out=args.json_out,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
