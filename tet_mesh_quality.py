#!/usr/bin/env python3
"""
Compute quality metrics for tetrahedral meshes stored in Gmsh .msh files.

Supported input:
- Gmsh 4.x ASCII or binary
- Gmsh 2.2 ASCII

For high-order tetrahedra, only the 4 corner nodes are used for the linear
quality metrics.
"""

from __future__ import annotations

import argparse
import json
import math
import struct
from pathlib import Path
from typing import BinaryIO


TET_NODE_COUNTS = {
    4: 4,    # tetrahedron
    11: 10,  # second-order tetrahedron
    29: 20,  # third-order tetrahedron
    30: 35,  # fourth-order tetrahedron
    31: 56,  # fifth-order tetrahedron
}


def vsub(a, b):
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def dot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def cross(a, b):
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def norm(a):
    return math.sqrt(dot(a, a))


def det3(a, b, c):
    return dot(a, cross(b, c))


def solve3(a, b):
    det_a = det3(a[0], a[1], a[2])
    if abs(det_a) < 1e-30:
        return None
    # Cramer's rule for a matrix stored as rows.
    col0 = (a[0][0], a[1][0], a[2][0])
    col1 = (a[0][1], a[1][1], a[2][1])
    col2 = (a[0][2], a[1][2], a[2][2])
    x = det3(b, col1, col2) / det_a
    y = det3(col0, b, col2) / det_a
    z = det3(col0, col1, b) / det_a
    return (x, y, z)


def face_area(a, b, c):
    return 0.5 * norm(cross(vsub(b, a), vsub(c, a)))


def percentile(sorted_values, p):
    if not sorted_values:
        return None
    if len(sorted_values) == 1:
        return sorted_values[0]
    pos = (len(sorted_values) - 1) * p
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return sorted_values[lo]
    frac = pos - lo
    return sorted_values[lo] * (1.0 - frac) + sorted_values[hi] * frac


def summarize(values):
    vals = sorted(v for v in values if math.isfinite(v))
    if not vals:
        return {
            "count": 0,
            "min": None,
            "p01": None,
            "p05": None,
            "median": None,
            "mean": None,
            "p95": None,
            "p99": None,
            "max": None,
        }
    return {
        "count": len(vals),
        "min": vals[0],
        "p01": percentile(vals, 0.01),
        "p05": percentile(vals, 0.05),
        "median": percentile(vals, 0.50),
        "mean": sum(vals) / len(vals),
        "p95": percentile(vals, 0.95),
        "p99": percentile(vals, 0.99),
        "max": vals[-1],
    }


def tetra_metrics(points):
    a, b, c, d = points
    ab = vsub(b, a)
    ac = vsub(c, a)
    ad = vsub(d, a)
    signed_jacobian = det3(ab, ac, ad)
    signed_volume = signed_jacobian / 6.0
    volume = abs(signed_volume)

    edge_pairs = ((a, b), (a, c), (a, d), (b, c), (b, d), (c, d))
    edges = [norm(vsub(q, p)) for p, q in edge_pairs]
    edge_sq_sum = sum(e * e for e in edges)
    min_edge = min(edges)
    max_edge = max(edges)

    surface_area = (
        face_area(a, b, c)
        + face_area(a, b, d)
        + face_area(a, c, d)
        + face_area(b, c, d)
    )
    inradius = 3.0 * volume / surface_area if surface_area > 0.0 else 0.0

    rhs = (dot(ab, ab) * 0.5, dot(ac, ac) * 0.5, dot(ad, ad) * 0.5)
    center_rel = solve3((ab, ac, ad), rhs)
    circumradius = norm(center_rel) if center_rel is not None else math.inf

    if volume > 0.0 and math.isfinite(circumradius) and circumradius > 0.0:
        radius_ratio = 3.0 * inradius / circumradius
        aspect_ratio_beta = circumradius / (3.0 * inradius) if inradius > 0.0 else math.inf
    else:
        radius_ratio = 0.0
        aspect_ratio_beta = math.inf

    srms = math.sqrt(edge_sq_sum / 6.0)
    aspect_ratio_gamma = (srms ** 3) / (8.479670 * volume) if volume > 0.0 else math.inf
    mean_ratio = 12.0 * ((3.0 * volume) ** (2.0 / 3.0)) / edge_sq_sum if volume > 0.0 and edge_sq_sum > 0.0 else 0.0
    edge_ratio = max_edge / min_edge if min_edge > 0.0 else math.inf

    corner_triples = (
        (vsub(b, a), vsub(c, a), vsub(d, a)),
        (vsub(a, b), vsub(d, b), vsub(c, b)),
        (vsub(a, c), vsub(b, c), vsub(d, c)),
        (vsub(a, d), vsub(c, d), vsub(b, d)),
    )
    scaled = []
    for u, v, w in corner_triples:
        denom = norm(u) * norm(v) * norm(w)
        scaled.append(det3(u, v, w) / denom if denom > 0.0 else -math.inf)
    scaled_jacobian = min(scaled)

    return {
        "signed_volume": signed_volume,
        "volume": volume,
        "min_edge": min_edge,
        "max_edge": max_edge,
        "edge_ratio": edge_ratio,
        "inradius": inradius,
        "circumradius": circumradius,
        "radius_ratio": radius_ratio,
        "aspect_ratio_beta": aspect_ratio_beta,
        "aspect_ratio_gamma": aspect_ratio_gamma,
        "mean_ratio": mean_ratio,
        "scaled_jacobian": scaled_jacobian,
    }


def read_line(f: BinaryIO) -> bytes:
    line = f.readline()
    if not line:
        raise EOFError("Unexpected end of file")
    return line.rstrip(b"\r\n")


def skip_section(f: BinaryIO, section_name: bytes):
    end = b"$End" + section_name[1:]
    while True:
        line = read_line(f)
        if line == end:
            return


def read_exact(f: BinaryIO, n: int) -> bytes:
    data = f.read(n)
    if len(data) != n:
        raise EOFError("Unexpected end of file while reading binary block")
    return data


def consume_end_section(f: BinaryIO, name: bytes):
    expected = b"$End" + name[1:]
    while True:
        line = read_line(f)
        if line:
            if line != expected:
                raise ValueError(f"Expected {expected.decode()}, got {line!r}")
            return


class BinaryReader:
    def __init__(self, f: BinaryIO, size_t_bytes: int):
        if size_t_bytes not in (4, 8):
            raise ValueError(f"Unsupported size_t byte width: {size_t_bytes}")
        self.f = f
        self.size_t_fmt = "<Q" if size_t_bytes == 8 else "<I"
        self.size_t_bytes = size_t_bytes

    def int32(self):
        return struct.unpack("<i", read_exact(self.f, 4))[0]

    def size_t(self):
        return struct.unpack(self.size_t_fmt, read_exact(self.f, self.size_t_bytes))[0]

    def double3(self):
        return struct.unpack("<ddd", read_exact(self.f, 24))


def parse_msh4_binary(path: Path, size_t_bytes: int):
    nodes = {}
    tets = []
    with path.open("rb") as f:
        while True:
            line = f.readline()
            if not line:
                break
            line = line.rstrip(b"\r\n")
            if line == b"$MeshFormat":
                read_line(f)
                read_exact(f, 4)
                consume_end_section(f, b"$MeshFormat")
            elif line == b"$Nodes":
                br = BinaryReader(f, size_t_bytes)
                num_blocks = br.size_t()
                br.size_t()
                br.size_t()
                br.size_t()
                for _ in range(num_blocks):
                    br.int32()
                    br.int32()
                    parametric = br.int32()
                    count = br.size_t()
                    tags = [br.size_t() for _ in range(count)]
                    for tag in tags:
                        nodes[tag] = br.double3()
                        if parametric:
                            raise ValueError("Parametric Gmsh nodes are not supported")
                consume_end_section(f, b"$Nodes")
            elif line == b"$Elements":
                br = BinaryReader(f, size_t_bytes)
                num_blocks = br.size_t()
                br.size_t()
                br.size_t()
                br.size_t()
                for _ in range(num_blocks):
                    br.int32()
                    br.int32()
                    element_type = br.int32()
                    count = br.size_t()
                    node_count = msh4_element_node_count(element_type)
                    if node_count is None:
                        for _ in range(count):
                            br.size_t()
                            raise ValueError(f"Unsupported element type in binary block: {element_type}")
                    for _ in range(count):
                        br.size_t()
                        conn = [br.size_t() for _ in range(node_count)]
                        if element_type in TET_NODE_COUNTS:
                            tets.append(tuple(conn[:4]))
                consume_end_section(f, b"$Elements")
            elif line.startswith(b"$"):
                skip_section(f, line)
    return nodes, tets


def msh4_element_node_count(element_type: int):
    counts = {
        1: 2, 2: 3, 3: 4, 4: 4, 5: 8, 6: 6, 7: 5, 8: 3, 9: 6, 10: 9, 11: 10,
        12: 27, 13: 18, 14: 14, 15: 1, 16: 8, 17: 20, 18: 15, 19: 13, 21: 10,
        23: 15, 25: 21, 26: 4, 27: 5, 28: 6, 29: 20, 30: 35, 31: 56,
    }
    return counts.get(element_type)


def parse_msh_ascii(path: Path, version: str):
    nodes = {}
    tets = []
    lines = path.read_text(errors="replace").splitlines()
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if line == "$Nodes":
            i += 1
            header = lines[i].split()
            if version.startswith("4."):
                num_blocks = int(header[0])
                i += 1
                for _ in range(num_blocks):
                    entity_dim, entity_tag, parametric, count = map(int, lines[i].split())
                    if parametric:
                        raise ValueError("Parametric Gmsh nodes are not supported")
                    i += 1
                    tags = [int(lines[i + j].split()[0]) for j in range(count)]
                    i += count
                    for tag in tags:
                        vals = lines[i].split()
                        nodes[tag] = (float(vals[0]), float(vals[1]), float(vals[2]))
                        i += 1
            else:
                count = int(header[0])
                i += 1
                for _ in range(count):
                    vals = lines[i].split()
                    nodes[int(vals[0])] = (float(vals[1]), float(vals[2]), float(vals[3]))
                    i += 1
        elif line == "$Elements":
            i += 1
            header = lines[i].split()
            if version.startswith("4."):
                num_blocks = int(header[0])
                i += 1
                for _ in range(num_blocks):
                    entity_dim, entity_tag, element_type, count = map(int, lines[i].split())
                    node_count = msh4_element_node_count(element_type)
                    if node_count is None:
                        raise ValueError(f"Unsupported element type: {element_type}")
                    i += 1
                    for _ in range(count):
                        vals = list(map(int, lines[i].split()))
                        if element_type in TET_NODE_COUNTS:
                            tets.append(tuple(vals[1:5]))
                        i += 1
            else:
                count = int(header[0])
                i += 1
                for _ in range(count):
                    vals = list(map(int, lines[i].split()))
                    element_type = vals[1]
                    num_tags = vals[2]
                    conn = vals[3 + num_tags:]
                    if element_type in TET_NODE_COUNTS:
                        tets.append(tuple(conn[:4]))
                    i += 1
        else:
            i += 1
    return nodes, tets


def read_msh(path: Path):
    with path.open("rb") as f:
        if read_line(f) != b"$MeshFormat":
            raise ValueError("Not a Gmsh .msh file: missing $MeshFormat")
        fmt = read_line(f).decode("ascii", errors="replace").split()
        version = fmt[0]
        binary = int(fmt[1])
        size_t_bytes = int(fmt[2]) if len(fmt) > 2 else 8
    if binary:
        if not version.startswith("4."):
            raise ValueError("Binary Gmsh 2.x is not supported; use Gmsh 4.x binary or ASCII")
        return parse_msh4_binary(path, size_t_bytes)
    return parse_msh_ascii(path, version)


def format_number(x):
    if x is None:
        return "n/a"
    if not math.isfinite(float(x)):
        return str(x)
    return f"{float(x):.6g}"


def main():
    parser = argparse.ArgumentParser(description="Compute tetrahedral mesh quality metrics for a Gmsh .msh file.")
    parser.add_argument("msh", type=Path, help="Input .msh file")
    parser.add_argument("--json-out", type=Path, help="Optional JSON summary path")
    parser.add_argument("--bad-scaled-jacobian", type=float, default=0.2, help="Threshold for poor scaled Jacobian")
    parser.add_argument("--bad-mean-ratio", type=float, default=0.2, help="Threshold for poor mean ratio")
    parser.add_argument("--bad-radius-ratio", type=float, default=0.2, help="Threshold for poor normalized radius ratio")
    args = parser.parse_args()

    nodes, tets = read_msh(args.msh)
    if not tets:
        raise SystemExit("No tetrahedral elements found")

    metric_values = {
        "signed_volume": [],
        "volume": [],
        "min_edge": [],
        "max_edge": [],
        "edge_ratio": [],
        "inradius": [],
        "circumradius": [],
        "radius_ratio": [],
        "aspect_ratio_beta": [],
        "aspect_ratio_gamma": [],
        "mean_ratio": [],
        "scaled_jacobian": [],
    }

    negative = 0
    zero = 0
    missing_nodes = 0
    for conn in tets:
        try:
            pts = [nodes[tag] for tag in conn]
        except KeyError:
            missing_nodes += 1
            continue
        m = tetra_metrics(pts)
        if m["signed_volume"] < 0.0:
            negative += 1
        if m["volume"] <= 1e-30:
            zero += 1
        for key, val in m.items():
            metric_values[key].append(val)

    summary = {
        "input": str(args.msh),
        "num_nodes": len(nodes),
        "num_tetrahedra": len(tets),
        "num_evaluated_tetrahedra": len(metric_values["volume"]),
        "num_missing_node_tetrahedra": missing_nodes,
        "num_negative_signed_volume": negative,
        "num_near_zero_volume": zero,
        "thresholds": {
            "bad_scaled_jacobian_lt": args.bad_scaled_jacobian,
            "bad_mean_ratio_lt": args.bad_mean_ratio,
            "bad_radius_ratio_lt": args.bad_radius_ratio,
        },
        "metrics": {key: summarize(vals) for key, vals in metric_values.items()},
        "bad_counts": {
            "scaled_jacobian_lt_threshold": sum(v < args.bad_scaled_jacobian for v in metric_values["scaled_jacobian"]),
            "mean_ratio_lt_threshold": sum(v < args.bad_mean_ratio for v in metric_values["mean_ratio"]),
            "radius_ratio_lt_threshold": sum(v < args.bad_radius_ratio for v in metric_values["radius_ratio"]),
        },
    }

    print(f"Input: {args.msh}")
    print(f"Nodes: {summary['num_nodes']}")
    print(f"Tetrahedra: {summary['num_tetrahedra']} evaluated: {summary['num_evaluated_tetrahedra']}")
    print(f"Negative signed volume: {negative}")
    print(f"Near-zero volume: {zero}")
    print("")
    print("Metric                 min        p05        median     mean       p95        max")
    print("-" * 83)
    for key in (
        "volume",
        "min_edge",
        "edge_ratio",
        "inradius",
        "circumradius",
        "radius_ratio",
        "aspect_ratio_beta",
        "aspect_ratio_gamma",
        "mean_ratio",
        "scaled_jacobian",
    ):
        s = summary["metrics"][key]
        print(
            f"{key:<22}"
            f"{format_number(s['min']):>10} "
            f"{format_number(s['p05']):>10} "
            f"{format_number(s['median']):>10} "
            f"{format_number(s['mean']):>10} "
            f"{format_number(s['p95']):>10} "
            f"{format_number(s['max']):>10}"
        )
    print("")
    print(
        "Bad counts: "
        f"scaled_jacobian < {args.bad_scaled_jacobian}: {summary['bad_counts']['scaled_jacobian_lt_threshold']}, "
        f"mean_ratio < {args.bad_mean_ratio}: {summary['bad_counts']['mean_ratio_lt_threshold']}, "
        f"radius_ratio < {args.bad_radius_ratio}: {summary['bad_counts']['radius_ratio_lt_threshold']}"
    )

    if args.json_out:
        args.json_out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"JSON written: {args.json_out}")


if __name__ == "__main__":
    main()
