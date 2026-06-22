import torch

from scene.hierarchical_tetrahedra_model import HierarchicalTetrahedraModel


def _infer_tets_device(tets):
    if hasattr(tets, "_xyz") and tets._xyz is not None:
        return tets._xyz.device
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _get_axis_index(viewpoint_camera):
    axis_name = str(viewpoint_camera.image_path).split("/")[-1].split("\\")[-1].lower()
    mapping = {"x": 0, "y": 1, "z": 2}
    return mapping.get(axis_name, 2)


def _has_explicit_plane_geometry(viewpoint_camera):
    required = (
        "plane_origin_world",
        "plane_normal_world",
        "plane_x_axis_world",
        "plane_y_axis_world",
        "plane_x_step",
        "plane_y_step",
    )
    return all(getattr(viewpoint_camera, key, None) is not None for key in required)


def _get_plane_geometry(viewpoint_camera, device, dtype):
    plane_origin = torch.as_tensor(viewpoint_camera.plane_origin_world, device=device, dtype=dtype)
    plane_normal = torch.as_tensor(viewpoint_camera.plane_normal_world, device=device, dtype=dtype)
    plane_x_axis = torch.as_tensor(viewpoint_camera.plane_x_axis_world, device=device, dtype=dtype)
    plane_y_axis = torch.as_tensor(viewpoint_camera.plane_y_axis_world, device=device, dtype=dtype)
    plane_x_step = torch.as_tensor(float(viewpoint_camera.plane_x_step), device=device, dtype=dtype)
    plane_y_step = torch.as_tensor(float(viewpoint_camera.plane_y_step), device=device, dtype=dtype)
    return plane_origin, plane_normal, plane_x_axis, plane_y_axis, plane_x_step, plane_y_step


def _resolve_slice_bounds(viewpoint_camera, tets, lod):
    device = _infer_tets_device(tets)
    if getattr(viewpoint_camera, "slice_bounds_min", None) is not None:
        slice_bounds_min = torch.as_tensor(viewpoint_camera.slice_bounds_min, device=device, dtype=torch.float32)
        slice_bounds_max = torch.as_tensor(viewpoint_camera.slice_bounds_max, device=device, dtype=torch.float32)
        return slice_bounds_min, slice_bounds_max
    vertices = tets.get_vertices(lod)
    return vertices.min(dim=0).values, vertices.max(dim=0).values


def _build_screenspace_points(tets, num_cells):
    screenspace_points = torch.zeros(
        (num_cells, 4),
        dtype=tets._xyz.dtype,
        requires_grad=True,
        device=tets._xyz.device,
    )
    try:
        screenspace_points.retain_grad()
    except RuntimeError:
        pass
    return screenspace_points


def _build_slice_intersection_mask(viewpoint_camera, tets, lod, cells, eps=1e-6):
    vertices = tets.get_vertices(lod)
    if _has_explicit_plane_geometry(viewpoint_camera):
        plane_origin, plane_normal, _, _, _, _ = _get_plane_geometry(viewpoint_camera, vertices.device, vertices.dtype)
        signed_dist = (vertices - plane_origin.unsqueeze(0)) @ plane_normal
        tet_signed = signed_dist[cells]
        tet_min = tet_signed.min(dim=1).values
        tet_max = tet_signed.max(dim=1).values
        return (tet_min <= eps) & (tet_max >= -eps)
    axis_idx = _get_axis_index(viewpoint_camera)
    plane_coord = torch.as_tensor(viewpoint_camera.camera_center, device=vertices.device, dtype=vertices.dtype)[axis_idx]
    cell_vertices_axis = vertices[cells][:, :, axis_idx]
    cell_min = cell_vertices_axis.min(dim=1).values
    cell_max = cell_vertices_axis.max(dim=1).values
    return (cell_min <= plane_coord + eps) & (cell_max >= plane_coord - eps)


def _covariance_to_matrix(cov3d_precomp):
    sigma = torch.zeros((cov3d_precomp.shape[0], 3, 3), device=cov3d_precomp.device, dtype=cov3d_precomp.dtype)
    sigma[:, 0, 0] = cov3d_precomp[:, 0]
    sigma[:, 1, 0] = cov3d_precomp[:, 1]
    sigma[:, 0, 1] = cov3d_precomp[:, 1]
    sigma[:, 2, 0] = cov3d_precomp[:, 2]
    sigma[:, 0, 2] = cov3d_precomp[:, 2]
    sigma[:, 1, 1] = cov3d_precomp[:, 3]
    sigma[:, 2, 1] = cov3d_precomp[:, 4]
    sigma[:, 1, 2] = cov3d_precomp[:, 4]
    sigma[:, 2, 2] = cov3d_precomp[:, 5]
    return sigma


def _full_visibility_mask(num_cells, selected_indices, active_mask):
    visibility = torch.zeros((num_cells,), dtype=torch.bool, device=active_mask.device)
    visibility[selected_indices] = active_mask
    return visibility


def _prepare_occ_tensors(viewpoint_camera, tets, lod):
    cells = tets.get_cells(lod)
    screenspace_points = _build_screenspace_points(tets, cells.shape[0])
    means3d, cov3d_precomp = tets.get_gs_mean_covs_uniform(lod)
    opacities = tets.get_opacities(lod)
    if torch.isnan(means3d).any() or torch.isinf(means3d).any():
        raise ValueError("NaN or Inf found in means3d before occupancy rendering")
    if torch.isnan(cov3d_precomp).any() or torch.isinf(cov3d_precomp).any():
        raise ValueError("NaN or Inf found in cov3d_precomp before occupancy rendering")
    if torch.isnan(opacities).any() or torch.isinf(opacities).any():
        raise ValueError("NaN or Inf found in opacities before occupancy rendering")
    return cells, screenspace_points, means3d, cov3d_precomp, opacities


def _pixel_center_plane_grid(viewpoint_camera, slice_bounds_min, slice_bounds_max, device, dtype):
    width = int(viewpoint_camera.image_width)
    height = int(viewpoint_camera.image_height)
    xs = torch.arange(width, device=device, dtype=dtype)
    ys = torch.arange(height, device=device, dtype=dtype)
    grid_x, grid_y = torch.meshgrid(xs, ys, indexing="xy")
    plane_x = slice_bounds_min[0] + ((2.0 * grid_x + 1.0) / float(width)) * 0.5 * (slice_bounds_max[0] - slice_bounds_min[0])
    plane_y = slice_bounds_min[1] + ((2.0 * grid_y + 1.0) / float(height)) * 0.5 * (slice_bounds_max[1] - slice_bounds_min[1])
    return grid_x, grid_y, plane_x, plane_y


def _pixel_center_plane_grid_explicit(viewpoint_camera, device, dtype):
    width = int(viewpoint_camera.image_width)
    height = int(viewpoint_camera.image_height)
    xs = torch.arange(width, device=device, dtype=dtype)
    ys = torch.arange(height, device=device, dtype=dtype)
    return torch.meshgrid(xs, ys, indexing="xy")


def _conditional_slice_params_torch(means3d, cov3d_precomp, plane_coord, axis_idx):
    sigma_xx = cov3d_precomp[:, 0]
    sigma_xy = cov3d_precomp[:, 1]
    sigma_xz = cov3d_precomp[:, 2]
    sigma_yy = cov3d_precomp[:, 3]
    sigma_yz = cov3d_precomp[:, 4]
    sigma_zz = cov3d_precomp[:, 5]

    if axis_idx == 0:
        t_minus_mu = plane_coord - means3d[:, 0]
        mu_u = means3d[:, 1] + sigma_xy * t_minus_mu / (sigma_xx + 1e-8)
        mu_v = means3d[:, 2] + sigma_xz * t_minus_mu / (sigma_xx + 1e-8)
        cov_00 = sigma_yy - (sigma_xy * sigma_xy) / (sigma_xx + 1e-6)
        cov_01 = sigma_yz - (sigma_xy * sigma_xz) / (sigma_xx + 1e-6)
        cov_11 = sigma_zz - (sigma_xz * sigma_xz) / (sigma_xx + 1e-6)
        slice_coord = means3d[:, 0]
        sigma_axis = sigma_xx
    elif axis_idx == 1:
        t_minus_mu = plane_coord - means3d[:, 1]
        mu_u = means3d[:, 0] + sigma_xy * t_minus_mu / (sigma_yy + 1e-8)
        mu_v = means3d[:, 2] + sigma_yz * t_minus_mu / (sigma_yy + 1e-8)
        cov_00 = sigma_xx - (sigma_xy * sigma_xy) / (sigma_yy + 1e-6)
        cov_01 = sigma_xz - (sigma_xy * sigma_yz) / (sigma_yy + 1e-6)
        cov_11 = sigma_zz - (sigma_yz * sigma_yz) / (sigma_yy + 1e-6)
        slice_coord = means3d[:, 1]
        sigma_axis = sigma_yy
    else:
        t_minus_mu = plane_coord - means3d[:, 2]
        mu_u = means3d[:, 0] + sigma_xz * t_minus_mu / (sigma_zz + 1e-8)
        mu_v = means3d[:, 1] + sigma_yz * t_minus_mu / (sigma_zz + 1e-8)
        cov_00 = sigma_xx - (sigma_xz * sigma_xz) / (sigma_zz + 1e-6)
        cov_01 = sigma_xy - (sigma_xz * sigma_yz) / (sigma_zz + 1e-6)
        cov_11 = sigma_yy - (sigma_yz * sigma_yz) / (sigma_zz + 1e-6)
        slice_coord = means3d[:, 2]
        sigma_axis = sigma_zz

    cov2d = torch.stack(
        [
            torch.stack([cov_00, cov_01], dim=-1),
            torch.stack([cov_01, cov_11], dim=-1),
        ],
        dim=-2,
    )
    return mu_u, mu_v, cov2d, slice_coord, sigma_axis


def _conditional_slice_params_explicit_torch(means3d, cov3d_precomp, plane_origin, plane_normal, plane_x_axis, plane_y_axis):
    basis = torch.stack([plane_x_axis, plane_y_axis, plane_normal], dim=1)
    rel_means = means3d - plane_origin.unsqueeze(0)
    means_local = rel_means @ basis

    sigma_world = _covariance_to_matrix(cov3d_precomp)
    sigma_local = torch.einsum("ji,njk,kl->nil", basis, sigma_world, basis)

    sigma_uu = sigma_local[:, 0, 0]
    sigma_uv = sigma_local[:, 0, 1]
    sigma_un = sigma_local[:, 0, 2]
    sigma_vv = sigma_local[:, 1, 1]
    sigma_vn = sigma_local[:, 1, 2]
    sigma_nn = sigma_local[:, 2, 2]

    t_minus_mu = -means_local[:, 2]
    mu_u = means_local[:, 0] + sigma_un * t_minus_mu / (sigma_nn + 1e-8)
    mu_v = means_local[:, 1] + sigma_vn * t_minus_mu / (sigma_nn + 1e-8)
    cov_00 = sigma_uu - (sigma_un * sigma_un) / (sigma_nn + 1e-6)
    cov_01 = sigma_uv - (sigma_un * sigma_vn) / (sigma_nn + 1e-6)
    cov_11 = sigma_vv - (sigma_vn * sigma_vn) / (sigma_nn + 1e-6)

    cov2d = torch.stack(
        [
            torch.stack([cov_00, cov_01], dim=-1),
            torch.stack([cov_01, cov_11], dim=-1),
        ],
        dim=-2,
    )
    slice_coord = means_local[:, 2]
    sigma_axis = sigma_nn
    return mu_u, mu_v, cov2d, slice_coord, sigma_axis


def _build_plane_points(plane_x, plane_y, plane_coord, axis_idx):
    if axis_idx == 0:
        return torch.stack([torch.full_like(plane_x, plane_coord), plane_x, plane_y], dim=-1)
    if axis_idx == 1:
        return torch.stack([plane_x, torch.full_like(plane_x, plane_coord), plane_y], dim=-1)
    return torch.stack([plane_x, plane_y, torch.full_like(plane_x, plane_coord)], dim=-1)


def _build_plane_points_explicit(grid_x, grid_y, plane_origin, plane_x_axis, plane_y_axis, plane_x_step, plane_y_step):
    return (
        plane_origin.view(1, 1, 3)
        + grid_x.unsqueeze(-1) * plane_x_step * plane_x_axis.view(1, 1, 3)
        + grid_y.unsqueeze(-1) * plane_y_step * plane_y_axis.view(1, 1, 3)
    )


def _aggregate_occupancy(contrib, mode, occ_thresh):
    if mode == "max":
        return torch.max(contrib, dim=0).values
    if mode == "sum":
        return torch.sum(contrib, dim=0)
    if mode == "sum_clip":
        return torch.clamp(torch.sum(contrib, dim=0), 0.0, 1.0)
    if mode == "prob_union":
        return 1.0 - torch.prod(torch.clamp(1.0 - contrib, 0.0, 1.0), dim=0)
    if mode == "bin":
        return (torch.max(contrib, dim=0).values >= occ_thresh).to(contrib.dtype)
    raise ValueError(f"Unknown occupancy mode: {mode}")
def render_htet_occ_torch(
    viewpoint_camera,
    tets: HierarchicalTetrahedraModel,
    lod=0,
    mode="sum_clip",
    multiply_opacity=False,
    occ_thresh=1.0 / 255.0,
):
    slice_bounds_min, slice_bounds_max = _resolve_slice_bounds(viewpoint_camera, tets, lod)
    cells, screenspace_points, means3d, cov3d_precomp, opacities = _prepare_occ_tensors(viewpoint_camera, tets, lod)
    slice_mask = _build_slice_intersection_mask(viewpoint_camera, tets, lod, cells)
    selected_indices = torch.arange(cells.shape[0], device=means3d.device)[slice_mask]

    height = int(viewpoint_camera.image_height)
    width = int(viewpoint_camera.image_width)
    if selected_indices.numel() == 0:
        empty = torch.zeros((1, height, width), device=means3d.device, dtype=means3d.dtype)
        return {
            "render": empty,
            "viewspace_points": screenspace_points,
            "visibility_filter": torch.zeros((cells.shape[0],), dtype=torch.bool, device=means3d.device),
            "radii": torch.zeros((0,), dtype=torch.int32, device=means3d.device),
            "selected_indices": selected_indices,
        }

    means3d = means3d[slice_mask]
    cov3d_precomp = cov3d_precomp[slice_mask]
    opacities = opacities[slice_mask]

    device = means3d.device
    dtype = means3d.dtype
    if _has_explicit_plane_geometry(viewpoint_camera):
        plane_origin, plane_normal, plane_x_axis, plane_y_axis, plane_x_step, plane_y_step = _get_plane_geometry(
            viewpoint_camera,
            device,
            dtype,
        )
        grid_x, grid_y = _pixel_center_plane_grid_explicit(viewpoint_camera, device, dtype)
        plane_pts = _build_plane_points_explicit(
            grid_x,
            grid_y,
            plane_origin,
            plane_x_axis,
            plane_y_axis,
            plane_x_step,
            plane_y_step,
        )
        mu_u, mu_v, cov2d, slice_coord, sigma_axis = _conditional_slice_params_explicit_torch(
            means3d,
            cov3d_precomp,
            plane_origin,
            plane_normal,
            plane_x_axis,
            plane_y_axis,
        )
        sx = 1.0 / torch.clamp_min(plane_x_step, 1e-8)
        sy = 1.0 / torch.clamp_min(plane_y_step, 1e-8)
        px = mu_u * sx
        py = mu_v * sy
        plane_coord = torch.tensor(0.0, device=device, dtype=dtype)
    else:
        axis_idx = _get_axis_index(viewpoint_camera)
        plane_coord = torch.as_tensor(viewpoint_camera.camera_center, device=device, dtype=dtype)[axis_idx]
        grid_x, grid_y, plane_x, plane_y = _pixel_center_plane_grid(
            viewpoint_camera,
            slice_bounds_min,
            slice_bounds_max,
            device,
            dtype,
        )
        plane_pts = _build_plane_points(plane_x, plane_y, plane_coord, axis_idx)
        mu_u, mu_v, cov2d, slice_coord, sigma_axis = _conditional_slice_params_torch(
            means3d,
            cov3d_precomp,
            plane_coord,
            axis_idx,
        )
        sx = (width - 1) / torch.clamp_min(slice_bounds_max[0] - slice_bounds_min[0], 1e-8)
        sy = (height - 1) / torch.clamp_min(slice_bounds_max[1] - slice_bounds_min[1], 1e-8)
        px = (mu_u - slice_bounds_min[0]) / torch.clamp_min(slice_bounds_max[0] - slice_bounds_min[0], 1e-8) * (width - 1)
        py = (mu_v - slice_bounds_min[1]) / torch.clamp_min(slice_bounds_max[1] - slice_bounds_min[1], 1e-8) * (height - 1)

    cov2d_pix = cov2d.clone()
    cov2d_pix[:, 0, 0] = cov2d[:, 0, 0] * (sx * sx) + 0.3
    cov2d_pix[:, 0, 1] = cov2d[:, 0, 1] * (sx * sy)
    cov2d_pix[:, 1, 0] = cov2d_pix[:, 0, 1]
    cov2d_pix[:, 1, 1] = cov2d[:, 1, 1] * (sy * sy) + 0.3

    eigvals = torch.linalg.eigvalsh(cov2d_pix)
    base_radius = 1.4 * torch.sqrt(torch.clamp_min(eigvals[:, 1], 0.0))
    dist = torch.abs(plane_coord - slice_coord)
    p_z = torch.exp(-(dist * dist) / (sigma_axis + 1e-6))
    weighted_radius = base_radius * torch.sqrt(p_z)

    inv_cov = torch.linalg.inv(_covariance_to_matrix(cov3d_precomp))
    diff = plane_pts.unsqueeze(0) - means3d[:, None, None, :]
    power = -0.5 * torch.einsum("nhwi,nij,nhwj->nhw", diff, inv_cov, diff)
    contrib = torch.exp(power)
    if multiply_opacity:
        contrib = contrib * opacities[:, 0][:, None, None]

    bbox_mask = (
        (grid_x.unsqueeze(0) >= (px[:, None, None] - weighted_radius[:, None, None]))
        & (grid_x.unsqueeze(0) <= (px[:, None, None] + weighted_radius[:, None, None]))
        & (grid_y.unsqueeze(0) >= (py[:, None, None] - weighted_radius[:, None, None]))
        & (grid_y.unsqueeze(0) <= (py[:, None, None] + weighted_radius[:, None, None]))
        & (weighted_radius[:, None, None] >= 1.0)
    )
    contrib = contrib * bbox_mask.to(dtype)
    occ = _aggregate_occupancy(contrib, mode, occ_thresh)

    active_mask = weighted_radius >= 1.0
    visibility_filter = _full_visibility_mask(cells.shape[0], selected_indices, active_mask)
    return {
        "render": occ.unsqueeze(0),
        "viewspace_points": screenspace_points,
        "visibility_filter": visibility_filter,
        "radii": torch.ceil(torch.clamp_min(weighted_radius, 0.0)).to(torch.int32),
        "selected_indices": selected_indices,
    }


def render_htet_occ_cpu(
    viewpoint_camera,
    tets: HierarchicalTetrahedraModel,
    lod=0,
    mode="sum_clip",
    multiply_opacity=False,
    occ_thresh=1.0 / 255.0,
    radius_scale=1.4,
    min_radius=1.0,
):
    cells, _, means3d, cov3d_precomp, opacities = _prepare_occ_tensors(viewpoint_camera, tets, lod)
    slice_mask = _build_slice_intersection_mask(viewpoint_camera, tets, lod, cells)
    selected_indices = torch.arange(cells.shape[0], device=means3d.device)[slice_mask]

    height = int(viewpoint_camera.image_height)
    width = int(viewpoint_camera.image_width)
    empty = torch.zeros((1, height, width), dtype=torch.float32)
    if selected_indices.numel() == 0:
        return {
            "render": empty,
            "viewspace_points": torch.zeros((cells.shape[0], 4), dtype=torch.float32, requires_grad=True),
            "visibility_filter": torch.zeros((cells.shape[0],), dtype=torch.bool),
            "radii": torch.zeros((0,), dtype=torch.int32),
            "selected_indices": selected_indices.detach().cpu(),
        }

    slice_bounds_min, slice_bounds_max = _resolve_slice_bounds(viewpoint_camera, tets, lod)

    means3d = means3d[slice_mask].detach().cpu().to(torch.float32)
    cov3d_precomp = cov3d_precomp[slice_mask].detach().cpu().to(torch.float32)
    opacities = opacities[slice_mask].detach().cpu().to(torch.float32)
    slice_bounds_min = slice_bounds_min.detach().cpu().to(torch.float32)
    slice_bounds_max = slice_bounds_max.detach().cpu().to(torch.float32)

    if _has_explicit_plane_geometry(viewpoint_camera):
        plane_origin, plane_normal, plane_x_axis, plane_y_axis, plane_x_step, plane_y_step = _get_plane_geometry(
            viewpoint_camera,
            torch.device("cpu"),
            torch.float32,
        )
        xs = torch.arange(width, dtype=torch.float32)
        ys = torch.arange(height, dtype=torch.float32)
        grid_x, grid_y = torch.meshgrid(xs, ys, indexing="xy")
        plane_pts = _build_plane_points_explicit(
            grid_x,
            grid_y,
            plane_origin,
            plane_x_axis,
            plane_y_axis,
            plane_x_step,
            plane_y_step,
        )
        mu_u, mu_v, cov2d, slice_coord, sigma_axis = _conditional_slice_params_explicit_torch(
            means3d,
            cov3d_precomp,
            plane_origin,
            plane_normal,
            plane_x_axis,
            plane_y_axis,
        )
        sx = 1.0 / torch.clamp_min(plane_x_step, 1e-8)
        sy = 1.0 / torch.clamp_min(plane_y_step, 1e-8)
        px = mu_u * sx
        py = mu_v * sy
        plane_coord = torch.tensor(0.0, dtype=torch.float32)
    else:
        axis_idx = _get_axis_index(viewpoint_camera)
        plane_coord = torch.as_tensor(viewpoint_camera.camera_center, device=means3d.device, dtype=means3d.dtype)[axis_idx]
        plane_coord = plane_coord.detach().cpu().to(torch.float32)
        xs = torch.arange(width, dtype=torch.float32)
        ys = torch.arange(height, dtype=torch.float32)
        grid_x, grid_y = torch.meshgrid(xs, ys, indexing="xy")
        plane_x = slice_bounds_min[0] + ((2.0 * grid_x + 1.0) / float(width)) * 0.5 * (slice_bounds_max[0] - slice_bounds_min[0])
        plane_y = slice_bounds_min[1] + ((2.0 * grid_y + 1.0) / float(height)) * 0.5 * (slice_bounds_max[1] - slice_bounds_min[1])
        plane_pts = _build_plane_points(plane_x, plane_y, plane_coord, axis_idx)
        mu_u, mu_v, cov2d, slice_coord, sigma_axis = _conditional_slice_params_torch(
            means3d,
            cov3d_precomp,
            plane_coord,
            axis_idx,
        )
        sx = (width - 1) / torch.clamp_min(slice_bounds_max[0] - slice_bounds_min[0], 1e-8)
        sy = (height - 1) / torch.clamp_min(slice_bounds_max[1] - slice_bounds_min[1], 1e-8)
        px = (mu_u - slice_bounds_min[0]) / torch.clamp_min(slice_bounds_max[0] - slice_bounds_min[0], 1e-8) * (width - 1)
        py = (mu_v - slice_bounds_min[1]) / torch.clamp_min(slice_bounds_max[1] - slice_bounds_min[1], 1e-8) * (height - 1)

    cov2d_pix = cov2d.clone()
    cov2d_pix[:, 0, 0] = cov2d[:, 0, 0] * (sx * sx) + 0.3
    cov2d_pix[:, 0, 1] = cov2d[:, 0, 1] * (sx * sy)
    cov2d_pix[:, 1, 0] = cov2d_pix[:, 0, 1]
    cov2d_pix[:, 1, 1] = cov2d[:, 1, 1] * (sy * sy) + 0.3

    eigvals = torch.linalg.eigvalsh(cov2d_pix)
    base_radius = radius_scale * torch.sqrt(torch.clamp_min(eigvals[:, 1], 0.0))
    dist = torch.abs(plane_coord - slice_coord)
    p_z = torch.exp(-(dist * dist) / (sigma_axis + 1e-6))
    weighted_radius = base_radius * torch.sqrt(p_z)

    occ_stack = []
    inv_cov = torch.linalg.inv(_covariance_to_matrix(cov3d_precomp))
    for idx in range(means3d.shape[0]):
        if weighted_radius[idx] < 1.0:
            continue
        bbox_mask = (
            (grid_x >= (px[idx] - weighted_radius[idx]))
            & (grid_x <= (px[idx] + weighted_radius[idx]))
            & (grid_y >= (py[idx] - weighted_radius[idx]))
            & (grid_y <= (py[idx] + weighted_radius[idx]))
        )
        if not bbox_mask.any():
            continue
        diff = plane_pts - means3d[idx]
        power = -0.5 * torch.einsum("hwi,ij,hwj->hw", diff, inv_cov[idx], diff)
        contrib = torch.exp(power) * bbox_mask.to(torch.float32)
        if multiply_opacity:
            contrib = contrib * opacities[idx, 0]
        occ_stack.append(contrib)

    if not occ_stack:
        occ = empty[0]
        active_radii = torch.zeros((0,), dtype=torch.int32)
    else:
        contribs = torch.stack(occ_stack, dim=0)
        occ = _aggregate_occupancy(contribs, mode, occ_thresh)
        active_radii = torch.ceil(torch.clamp_min(weighted_radius, 0.0)).to(torch.int32)

    visibility_filter = torch.zeros((cells.shape[0],), dtype=torch.bool)
    visibility_filter[selected_indices.detach().cpu()] = weighted_radius.detach().cpu() > 0.0
    return {
        "render": occ.unsqueeze(0),
        "viewspace_points": torch.zeros((cells.shape[0], 4), dtype=torch.float32, requires_grad=True),
        "visibility_filter": visibility_filter,
        "radii": active_radii,
        "selected_indices": selected_indices.detach().cpu(),
    }
