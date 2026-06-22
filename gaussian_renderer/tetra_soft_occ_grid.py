import math
from typing import Optional, Sequence

import torch

from gaussian_renderer.volume_grid import (
    _parse_block_size_zyx,
    _parse_grid_origin_xyz,
    _parse_spacing_xyz,
    _volume_shape_to_size_xyz,
)


def _resolve_soft_boundary_width(
    alpha: float,
    boundary_thresh: float,
    soft_boundary_width: Optional[float],
) -> float:
    if soft_boundary_width is not None:
        if soft_boundary_width < 0.0:
            raise ValueError("soft_boundary_width must be non-negative.")
        return float(soft_boundary_width)
    safe_alpha = max(float(alpha), 1e-8)
    safe_thresh = min(max(float(boundary_thresh), 1e-8), 0.499999)
    return float(math.log((1.0 - safe_thresh) / safe_thresh) / safe_alpha)


def _build_tetra_halfspaces(cell_vertices: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    face_vertex_ids = torch.tensor(
        [
            [1, 2, 3],
            [0, 3, 2],
            [0, 1, 3],
            [0, 2, 1],
        ],
        device=cell_vertices.device,
        dtype=torch.long,
    )
    opp_vertex_ids = torch.tensor([0, 1, 2, 3], device=cell_vertices.device, dtype=torch.long)

    face_vertices = cell_vertices[:, face_vertex_ids]
    face_origins = face_vertices[:, :, 0, :]
    edge_ab = face_vertices[:, :, 1, :] - face_origins
    edge_ac = face_vertices[:, :, 2, :] - face_origins
    raw_normals = torch.cross(edge_ab, edge_ac, dim=-1)

    opposite_vertices = cell_vertices[:, opp_vertex_ids, :]
    direction_to_inside = opposite_vertices - face_origins
    orientation_sign = torch.where(
        (raw_normals * direction_to_inside).sum(dim=-1, keepdim=True) >= 0.0,
        torch.ones_like(raw_normals[..., :1]),
        -torch.ones_like(raw_normals[..., :1]),
    )
    inward_normals = raw_normals * orientation_sign
    inward_normals = inward_normals / torch.clamp_min(inward_normals.norm(dim=-1, keepdim=True), 1e-12)
    face_scale = torch.sum((opposite_vertices - face_origins) * inward_normals, dim=-1, keepdim=True)
    face_scale = torch.clamp_min(face_scale, 1e-12)
    return face_origins, inward_normals, face_scale


def _accumulate_block_tetra_contrib(
    points_xyz: torch.Tensor,
    face_origins: torch.Tensor,
    inward_normals: torch.Tensor,
    face_scale: torch.Tensor,
    alpha: float,
    halfspace_bias: float,
    opacities: Optional[torch.Tensor],
    multiply_opacity: bool,
    mode: str,
    tetra_chunk_size: int,
    global_gate_thresh: Optional[float],
    global_gate_steepness: float,
) -> torch.Tensor:
    num_points = points_xyz.shape[0]
    dtype = points_xyz.dtype
    device = points_xyz.device

    if mode in {"sum", "sum_clip"}:
        accum = torch.zeros((num_points,), device=device, dtype=dtype)
    elif mode in {"max", "bin"}:
        accum = torch.zeros((num_points,), device=device, dtype=dtype)
    elif mode == "prob_union":
        accum = torch.ones((num_points,), device=device, dtype=dtype)
    else:
        raise ValueError(f"Unsupported tetra occupancy mode: {mode}")

    for start in range(0, face_origins.shape[0], tetra_chunk_size):
        stop = min(start + tetra_chunk_size, face_origins.shape[0])
        chunk_origins = face_origins[start:stop]
        chunk_normals = inward_normals[start:stop]
        chunk_face_scale = face_scale[start:stop]

        halfspaces = torch.sum(
            (points_xyz.unsqueeze(0).unsqueeze(2) - chunk_origins[:, None, :, :]) * chunk_normals[:, None, :, :],
            dim=-1,
        )
        halfspaces = halfspaces / chunk_face_scale[:, None, :, 0]
        contrib = torch.sigmoid(float(alpha) * (halfspaces + float(halfspace_bias))).prod(dim=-1)

        if multiply_opacity and opacities is not None:
            contrib = contrib * opacities[start:stop, 0][:, None]

        if mode in {"sum", "sum_clip"}:
            accum = accum + contrib.sum(dim=0)
        elif mode in {"max", "bin"}:
            accum = torch.maximum(accum, contrib.max(dim=0).values)
        else:
            accum = accum * torch.prod(torch.clamp(1.0 - contrib, 0.0, 1.0), dim=0)

    if mode == "sum_clip":
        accum = torch.clamp(accum, 0.0, 1.0)
    elif mode == "bin":
        accum = (accum >= 0.5).to(dtype)
    elif mode == "prob_union":
        accum = 1.0 - accum

    if global_gate_thresh is not None:
        accum = torch.sigmoid(float(global_gate_steepness) * (accum - float(global_gate_thresh)))
    return accum


def render_occ_volume_from_tetrahedra(
    vertices: torch.Tensor,
    cells: torch.Tensor,
    volume_shape_zyx: Sequence[int],
    voxel_spacing_xyz: Sequence[float],
    grid_origin_xyz: Optional[Sequence[float]] = None,
    opacities: Optional[torch.Tensor] = None,
    alpha: float = 32.0,
    halfspace_bias: float = 0.05,
    mode: str = "prob_union",
    multiply_opacity: bool = False,
    block_size: Sequence[int] = (8, 8, 8),
    tetra_chunk_size: int = 128,
    boundary_thresh: float = 1e-4,
    soft_boundary_width: Optional[float] = None,
    global_gate_thresh: Optional[float] = None,
    global_gate_steepness: float = 50.0,
) -> dict:
    if vertices.ndim != 2 or vertices.shape[-1] != 3:
        raise ValueError("vertices must have shape [N, 3].")
    if cells.ndim != 2 or cells.shape[-1] != 4:
        raise ValueError("cells must have shape [T, 4].")
    if tetra_chunk_size <= 0:
        raise ValueError("tetra_chunk_size must be positive.")
    if float(alpha) <= 0.0:
        raise ValueError("alpha must be positive.")

    device = vertices.device
    dtype = vertices.dtype
    spacing_xyz = _parse_spacing_xyz(voxel_spacing_xyz, device=device, dtype=dtype)
    origin_xyz = _parse_grid_origin_xyz(grid_origin_xyz, device=device, dtype=dtype)
    size_xyz = _volume_shape_to_size_xyz(volume_shape_zyx, device=device)
    block_size_z, block_size_y, block_size_x = _parse_block_size_zyx(block_size)
    block_size_xyz = torch.tensor([block_size_x, block_size_y, block_size_z], device=device, dtype=torch.long)
    grid_shape_xyz = torch.div(size_xyz + block_size_xyz - 1, block_size_xyz, rounding_mode="floor")

    cell_vertices = vertices[cells]
    face_origins, inward_normals, face_scale = _build_tetra_halfspaces(cell_vertices)
    effective_soft_width = _resolve_soft_boundary_width(
        alpha=alpha,
        boundary_thresh=boundary_thresh,
        soft_boundary_width=soft_boundary_width,
    )
    soft_width_xyz = torch.full((1, 3), float(effective_soft_width), device=device, dtype=dtype)

    bbox_min_xyz = cell_vertices.min(dim=1).values - soft_width_xyz
    bbox_max_xyz = cell_vertices.max(dim=1).values + soft_width_xyz
    bbox_min_idx_xyz = torch.floor((bbox_min_xyz - origin_xyz) / spacing_xyz).to(torch.long)
    bbox_max_idx_xyz = torch.ceil((bbox_max_xyz - origin_xyz) / spacing_xyz).to(torch.long)
    bbox_min_idx_xyz = torch.maximum(bbox_min_idx_xyz, torch.zeros_like(bbox_min_idx_xyz))
    bbox_max_idx_xyz = torch.minimum(bbox_max_idx_xyz, (size_xyz - 1).unsqueeze(0))

    active_mask = torch.all(bbox_min_idx_xyz <= bbox_max_idx_xyz, dim=1)
    selected_indices = torch.nonzero(active_mask, as_tuple=False).squeeze(-1)
    volume = torch.zeros(
        (int(volume_shape_zyx[0]), int(volume_shape_zyx[1]), int(volume_shape_zyx[2])),
        device=device,
        dtype=dtype,
    )

    if selected_indices.numel() == 0:
        visibility = torch.zeros((cells.shape[0],), dtype=torch.bool, device=device)
        return {
            "render": volume.unsqueeze(0),
            "visibility_filter": visibility,
            "selected_indices": selected_indices,
            "radii": torch.zeros((0,), dtype=torch.int32, device=device),
            "stats": {
                "num_tetrahedra": int(cells.shape[0]),
                "num_active_tetrahedra": 0,
                "num_nonempty_blocks": 0,
                "num_block_assignments": 0,
                "alpha": float(alpha),
                "soft_boundary_width": float(effective_soft_width),
            },
        }

    active_bbox_min = bbox_min_idx_xyz[selected_indices]
    active_bbox_max = bbox_max_idx_xyz[selected_indices]
    active_face_origins = face_origins[selected_indices]
    active_inward_normals = inward_normals[selected_indices]
    active_face_scale = face_scale[selected_indices]
    active_opacities = opacities[selected_indices] if opacities is not None else None

    block_min_xyz = torch.div(active_bbox_min, block_size_xyz, rounding_mode="floor")
    block_max_xyz = torch.div(active_bbox_max, block_size_xyz, rounding_mode="floor")
    block_span_xyz = block_max_xyz - block_min_xyz + 1
    block_count_per_tetra = torch.prod(block_span_xyz, dim=1)

    pair_offsets = torch.cumsum(block_count_per_tetra, dim=0) - block_count_per_tetra
    total_pairs = int(block_count_per_tetra.sum().item())
    pair_tetra_ids = torch.repeat_interleave(
        torch.arange(selected_indices.shape[0], device=device, dtype=torch.long),
        block_count_per_tetra,
    )
    local_linear = torch.arange(total_pairs, device=device, dtype=torch.long) - torch.repeat_interleave(
        pair_offsets,
        block_count_per_tetra,
    )

    pair_dims_xyz = block_span_xyz[pair_tetra_ids]
    yz_stride = pair_dims_xyz[:, 1] * pair_dims_xyz[:, 2]
    offset_x = torch.div(local_linear, yz_stride, rounding_mode="floor")
    remainder = local_linear - offset_x * yz_stride
    offset_y = torch.div(remainder, pair_dims_xyz[:, 2], rounding_mode="floor")
    offset_z = remainder - offset_y * pair_dims_xyz[:, 2]

    pair_block_xyz = block_min_xyz[pair_tetra_ids] + torch.stack([offset_x, offset_y, offset_z], dim=-1)
    grid_shape_x = int(grid_shape_xyz[0].item())
    grid_shape_y = int(grid_shape_xyz[1].item())
    block_linear = pair_block_xyz[:, 0] + grid_shape_x * (pair_block_xyz[:, 1] + grid_shape_y * pair_block_xyz[:, 2])

    order = torch.argsort(block_linear)
    sorted_block_ids = block_linear[order]
    sorted_tetra_ids = pair_tetra_ids[order]
    unique_block_ids, block_counts = torch.unique_consecutive(sorted_block_ids, return_counts=True)
    block_starts = torch.cumsum(block_counts, dim=0) - block_counts

    unique_block_ids_cpu = unique_block_ids.detach().cpu().tolist()
    block_counts_cpu = block_counts.detach().cpu().tolist()
    block_starts_cpu = block_starts.detach().cpu().tolist()
    grid_shape_z = int(grid_shape_xyz[2].item())

    for block_id, block_start, block_count in zip(unique_block_ids_cpu, block_starts_cpu, block_counts_cpu):
        bx = int(block_id % grid_shape_x)
        by = int((block_id // grid_shape_x) % grid_shape_y)
        bz = int(block_id // (grid_shape_x * grid_shape_y))
        if bz < 0 or bz >= grid_shape_z:
            continue

        x0 = bx * block_size_x
        x1 = min(x0 + block_size_x, int(volume_shape_zyx[2]))
        y0 = by * block_size_y
        y1 = min(y0 + block_size_y, int(volume_shape_zyx[1]))
        z0 = bz * block_size_z
        z1 = min(z0 + block_size_z, int(volume_shape_zyx[0]))

        x_coords = origin_xyz[0] + torch.arange(x0, x1, device=device, dtype=dtype) * spacing_xyz[0]
        y_coords = origin_xyz[1] + torch.arange(y0, y1, device=device, dtype=dtype) * spacing_xyz[1]
        z_coords = origin_xyz[2] + torch.arange(z0, z1, device=device, dtype=dtype) * spacing_xyz[2]

        grid_z, grid_y, grid_x = torch.meshgrid(z_coords, y_coords, x_coords, indexing="ij")
        points_xyz = torch.stack([grid_x, grid_y, grid_z], dim=-1).reshape(-1, 3)

        tetra_ids = sorted_tetra_ids[block_start : block_start + block_count]
        block_values = _accumulate_block_tetra_contrib(
            points_xyz=points_xyz,
            face_origins=active_face_origins[tetra_ids],
            inward_normals=active_inward_normals[tetra_ids],
            face_scale=active_face_scale[tetra_ids],
            alpha=alpha,
            halfspace_bias=halfspace_bias,
            opacities=None if active_opacities is None else active_opacities[tetra_ids],
            multiply_opacity=multiply_opacity,
            mode=mode,
            tetra_chunk_size=tetra_chunk_size,
            global_gate_thresh=global_gate_thresh,
            global_gate_steepness=global_gate_steepness,
        )
        volume[z0:z1, y0:y1, x0:x1] = block_values.view(z1 - z0, y1 - y0, x1 - x0)

    visibility = torch.zeros((cells.shape[0],), dtype=torch.bool, device=device)
    visibility[selected_indices] = True
    support_extent_vox = (active_bbox_max - active_bbox_min + 1).to(dtype=torch.float32)
    radii = torch.ceil(torch.clamp_min(0.5 * support_extent_vox.max(dim=1).values, 0.0)).to(torch.int32)

    return {
        "render": volume.unsqueeze(0),
        "visibility_filter": visibility,
        "selected_indices": selected_indices,
        "radii": radii,
        "stats": {
            "num_tetrahedra": int(cells.shape[0]),
            "num_active_tetrahedra": int(selected_indices.numel()),
            "num_nonempty_blocks": int(len(unique_block_ids_cpu)),
            "num_block_assignments": int(total_pairs),
            "alpha": float(alpha),
            "soft_boundary_width": float(effective_soft_width),
        },
    }


def render_htet_soft_occ_volume_torch(
    tets,
    volume_shape_zyx: Sequence[int],
    voxel_spacing_xyz: Sequence[float],
    grid_origin_xyz: Optional[Sequence[float]] = None,
    lod: int = 0,
    alpha: float = 32.0,
    halfspace_bias: float = 0.05,
    mode: str = "prob_union",
    multiply_opacity: bool = False,
    block_size: Sequence[int] = (8, 8, 8),
    tetra_chunk_size: int = 128,
    boundary_thresh: float = 1e-4,
    soft_boundary_width: Optional[float] = None,
    global_gate_thresh: Optional[float] = None,
    global_gate_steepness: float = 50.0,
) -> dict:
    vertices = tets.get_vertices(lod)
    cells = tets.get_cells(lod)
    opacities = tets.get_opacities(lod)
    return render_occ_volume_from_tetrahedra(
        vertices=vertices,
        cells=cells,
        volume_shape_zyx=volume_shape_zyx,
        voxel_spacing_xyz=voxel_spacing_xyz,
        grid_origin_xyz=grid_origin_xyz,
        opacities=opacities,
        alpha=alpha,
        halfspace_bias=halfspace_bias,
        mode=mode,
        multiply_opacity=multiply_opacity,
        block_size=block_size,
        tetra_chunk_size=tetra_chunk_size,
        boundary_thresh=boundary_thresh,
        soft_boundary_width=soft_boundary_width,
        global_gate_thresh=global_gate_thresh,
        global_gate_steepness=global_gate_steepness,
    )
