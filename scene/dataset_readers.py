import os
import json
from typing import NamedTuple, Optional

import meshio
import numpy as np
import torch
from PIL import Image

from scene.gaussian_model import BasicPointCloud, BasicTetrahedra
from utils.graphics_utils import getWorld2View2
from utils.mri_utils import (
    get_axis_image_size,
    get_axis_index,
    load_nifti_volume,
    map_slice_index_to_bounds,
    normalize_slice_to_rgb,
    resolve_nifti_path,
    world_to_local_sax,
)

# Axes to read from a multiaxis camera bundle.
# Edit this tuple manually, for example:
#   ("z",)
#   ("z", "y")
MULTIAXIS_READ_AXES = ("z", "y")
# Multi-axis camera merge mode: "interleave" or "sequential".
MULTIAXIS_MERGE_MODE = "interleave"


class CameraInfo(NamedTuple):
    uid: int
    R: np.array
    T: np.array
    FovY: np.array
    FovX: np.array
    image: np.array
    image_path: str
    image_name: str
    width: int
    height: int
    slice_bounds_min: np.array
    slice_bounds_max: np.array
    plane_origin_world: Optional[np.array] = None
    plane_normal_world: Optional[np.array] = None
    plane_x_axis_world: Optional[np.array] = None
    plane_y_axis_world: Optional[np.array] = None
    plane_x_step: Optional[float] = None
    plane_y_step: Optional[float] = None


class SceneInfo(NamedTuple):
    point_cloud: BasicPointCloud
    tetrahedra: BasicTetrahedra
    train_cameras: list
    test_cameras: list
    nerf_normalization: dict
    ply_path: Optional[str]


def getNerfppNorm(cam_info):
    def get_center_and_diag(cam_centers):
        cam_centers = np.hstack(cam_centers)
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        diagonal = np.max(dist)
        return center.flatten(), diagonal

    if not cam_info:
        return {"translate": np.zeros(3, dtype=np.float32), "radius": 1.0}

    cam_centers = []
    for cam in cam_info:
        w2c = getWorld2View2(cam.R, cam.T)
        c2w = np.linalg.inv(w2c)
        cam_centers.append(c2w[:3, 3:4])

    center, diagonal = get_center_and_diag(cam_centers)
    return {"translate": -center, "radius": max(float(diagonal) * 1.1, 1.0)}


def fetchPly(path):
    try:
        from plyfile import PlyData
    except ImportError:
        return None

    plydata = PlyData.read(path)
    vertices = plydata["vertex"]
    positions = np.vstack([vertices["x"], vertices["y"], vertices["z"]]).T
    colors = np.vstack([vertices["red"], vertices["green"], vertices["blue"]]).T / 255.0
    normals = np.vstack([vertices["nx"], vertices["ny"], vertices["nz"]]).T
    return BasicPointCloud(points=positions, colors=colors, normals=normals)


def fetchmsh(path):
    mesh = meshio.read(path)
    tetra_vertices = np.array(mesh.points, dtype=np.float32)
    tetra_cells = None
    for cell in mesh.cells:
        if cell.type == "tetra":
            tetra_cells = np.array(cell.data, dtype=np.int32)
            break
    if tetra_cells is None:
        raise ValueError(f"No tetrahedra cells found in {path}")
    return BasicTetrahedra(vertices=tetra_vertices, cells=tetra_cells, colors=None)


def storePly(path, xyz, rgb):
    normals = np.zeros_like(xyz)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    with open(path, "w", encoding="utf-8-sig") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {xyz.shape[0]}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property float nx\n")
        f.write("property float ny\n")
        f.write("property float nz\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")
        for row in attributes:
            f.write(
                f"{row[0]} {row[1]} {row[2]} {row[3]} {row[4]} {row[5]} "
                f"{int(row[6])} {int(row[7])} {int(row[8])}\n"
            )


def _fallback_camera_position(slice_index, axis_name, total_slices, scale):
    axis_idx = get_axis_index(axis_name)
    coord = torch.zeros(3, dtype=torch.float32)
    if total_slices <= 1:
        return coord

    normalized_pos = (float(slice_index) / float(total_slices - 1)) * 2.0 - 1.0
    if axis_idx == 0:
        coord[0] = normalized_pos * scale[0]
    elif axis_idx == 1:
        coord[1] = normalized_pos * scale[1]
    else:
        coord[2] = normalized_pos * scale[2]
    return coord


def _resolve_tetra_path(path):
    candidates = [
        os.path.join(path, "frame01_tetra.msh"),
        os.path.join(path, "frame01_tetra_sax_local.msh"),
        os.path.join(path, "cardiac_1.msh"),
        os.path.join(path, "points3d.msh"),
        os.path.join(path, "mesh", "frame01_tetra.msh"),
        os.path.join(path, "mesh", "time_01_tetra.msh"),
    ]
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    mesh_dir = os.path.join(path, "mesh")
    if os.path.isdir(mesh_dir):
        msh_files = sorted(
            [os.path.join(mesh_dir, name) for name in os.listdir(mesh_dir) if name.lower().endswith(".msh")]
        )
        if msh_files:
            return msh_files[0]
    return candidates[0]


def _load_or_create_tetrahedra(path, tetra_path, scale):
    if os.path.exists(tetra_path):
        return tetra_path, fetchmsh(tetra_path)

    print("Generating tetrahedra...")
    from utils.create_tetra_init import create_tetrahedral_grid

    grid_size = 30
    vertices, tetrahedra = create_tetrahedral_grid(scale, grid_size, grid_size, grid_size)
    output_path = os.path.join(path, "points3d.msh")
    meshio.Mesh(points=vertices, cells=[("tetra", tetrahedra)]).write(output_path, file_format="gmsh")
    print(f"Saved tetrahedra to {output_path}")
    return output_path, fetchmsh(output_path)


def _load_or_create_point_cloud(path, reference_vertices=None):
    if reference_vertices is not None and len(reference_vertices) > 0:
        xyz = np.asarray(reference_vertices, dtype=np.float32)
        rgb = np.full((xyz.shape[0], 3), 180, dtype=np.uint8)
    else:
        num_pts = 100_000
        print(f"Generating random point cloud ({num_pts})...")
        xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
        rgb = np.full((num_pts, 3), 180, dtype=np.uint8)
    normals = np.zeros_like(xyz, dtype=np.float32)
    pcd = BasicPointCloud(points=xyz, colors=rgb.astype(np.float32) / 255.0, normals=normals)
    return None, pcd


def _build_camera(
    idx,
    cam_pos,
    image,
    axis_name,
    image_name,
    slice_bounds_min,
    slice_bounds_max,
    plane_origin_world=None,
    plane_normal_world=None,
    plane_x_axis_world=None,
    plane_y_axis_world=None,
    plane_x_step=None,
    plane_y_step=None,
):
    r = np.eye(3, dtype=np.float32)
    t = -1.0 * np.asarray(cam_pos, dtype=np.float32)
    width, height = image.size
    return CameraInfo(
        uid=idx,
        R=r,
        T=t,
        FovY=np.pi / 2,
        FovX=np.pi / 2,
        image=image,
        image_path=axis_name,
        image_name=image_name,
        width=width,
        height=height,
        slice_bounds_min=np.asarray(slice_bounds_min, dtype=np.float32),
        slice_bounds_max=np.asarray(slice_bounds_max, dtype=np.float32),
        plane_origin_world=None if plane_origin_world is None else np.asarray(plane_origin_world, dtype=np.float32),
        plane_normal_world=None if plane_normal_world is None else np.asarray(plane_normal_world, dtype=np.float32),
        plane_x_axis_world=None if plane_x_axis_world is None else np.asarray(plane_x_axis_world, dtype=np.float32),
        plane_y_axis_world=None if plane_y_axis_world is None else np.asarray(plane_y_axis_world, dtype=np.float32),
        plane_x_step=None if plane_x_step is None else float(plane_x_step),
        plane_y_step=None if plane_y_step is None else float(plane_y_step),
    )


def _detect_plane_dataset(path):
    return os.path.exists(os.path.join(path, "plane_positions.json")) and os.path.isdir(
        os.path.join(path, "mask_slices_png")
    )


def _detect_camera_info_dataset(path):
    return os.path.exists(os.path.join(path, "camera_info.json")) and os.path.isdir(
        os.path.join(path, "mask_slices_png")
    )


def _detect_multiaxis_camera_dataset(path):
    found_axes = []
    requested_axes = MULTIAXIS_READ_AXES if MULTIAXIS_READ_AXES else ("x", "y", "z")
    for axis_name in requested_axes:
        axis_path = os.path.join(path, axis_name)
        if _detect_camera_info_dataset(axis_path):
            found_axes.append(axis_name)
    return found_axes


def _has_explicit_plane_geometry(record):
    required = (
        "plane_origin_world_xyz",
        "plane_x_axis_world_xyz",
        "plane_y_axis_world_xyz",
    )
    return all(key in record for key in required)


def _normalize_vector(vec):
    vec = np.asarray(vec, dtype=np.float32)
    norm = np.linalg.norm(vec)
    if norm < 1e-8:
        raise ValueError("Cannot normalize near-zero vector")
    return vec / norm


def _get_camera_plane_coordinate(cam_info, axis_name):
    axis_name = str(axis_name).lower()
    if cam_info.plane_origin_world is not None and cam_info.plane_normal_world is not None:
        plane_origin = np.asarray(cam_info.plane_origin_world, dtype=np.float32)
        plane_normal = _normalize_vector(cam_info.plane_normal_world)
        return float(np.dot(plane_origin, plane_normal))

    axis_idx = {"x": 0, "y": 1, "z": 2}.get(axis_name, 2)
    cam_pos = -np.asarray(cam_info.T, dtype=np.float32)
    return float(cam_pos[axis_idx])


def _estimate_axis_spacing(cam_infos, axis_name):
    coords = []
    for cam_info in cam_infos:
        cam_axis = str(cam_info.image_path).split("/")[-1].split("\\")[-1].lower()
        if cam_axis != str(axis_name).lower():
            continue
        coords.append(_get_camera_plane_coordinate(cam_info, axis_name))

    if len(coords) < 2:
        return None

    sorted_coords = np.sort(np.asarray(coords, dtype=np.float32))
    diffs = np.diff(sorted_coords)
    positive_diffs = diffs[diffs > 1e-6]
    if positive_diffs.size == 0:
        return None
    return float(np.median(positive_diffs))


def _build_local_frame_from_plane_records(records):
    plane_normal_world = _normalize_vector(records[0]["plane_normal_world"])
    reference = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    if abs(float(np.dot(reference, plane_normal_world))) > 0.95:
        reference = np.array([0.0, 1.0, 0.0], dtype=np.float32)

    x_axis_world = np.cross(reference, plane_normal_world)
    x_axis_world = _normalize_vector(x_axis_world)
    y_axis_world = np.cross(plane_normal_world, x_axis_world)
    y_axis_world = _normalize_vector(y_axis_world)
    direction = np.stack([x_axis_world, y_axis_world, plane_normal_world], axis=1).astype(np.float32)

    first_origin_world = np.asarray(records[0]["plane_origin_world"], dtype=np.float32)
    first_offset = float(records[0].get("plane_offset_mm_local_z", 0.0))
    origin = first_origin_world - direction[:, 2] * first_offset
    inv_direction = np.linalg.inv(direction).astype(np.float32)
    return origin.astype(np.float32), direction, inv_direction


def _find_sax_nifti_from_metadata(path):
    metadata_path = os.path.join(path, "metadata.json")
    if os.path.exists(metadata_path):
        try:
            with open(metadata_path, "r", encoding="utf-8-sig") as f:
                metadata = json.load(f)
            sax_path = metadata.get("sax")
            if sax_path and os.path.exists(sax_path):
                return sax_path
        except Exception:
            pass

    dataset_name = os.path.basename(os.path.normpath(path)).lower()
    inferred_candidates = []
    if "_t" in dataset_name:
        time_token = dataset_name.split("_t")[-1]
        inferred_candidates.extend(
            [
                os.path.join(os.path.dirname(path), "..", "v1_nifti", "SAX", f"time_{int(time_token):02d}.nii.gz"),
                os.path.join(os.path.dirname(path), "..", "v1_nifti", "SAX", "time_01.nii.gz"),
            ]
        )

    for candidate in inferred_candidates:
        candidate = os.path.abspath(candidate)
        if os.path.exists(candidate):
            return candidate
    return None


def _resolve_plane_dataset_frame(path, records):
    sax_nifti_path = _find_sax_nifti_from_metadata(path)
    if sax_nifti_path is not None:
        volume = load_nifti_volume(sax_nifti_path)
        return {
            "origin": np.asarray(volume["origin"], dtype=np.float32),
            "direction": np.asarray(volume["direction"], dtype=np.float32),
            "inv_direction": np.asarray(volume["inv_direction"], dtype=np.float32),
            "spacing": np.asarray(volume["spacing"], dtype=np.float32),
            "size_xyz": np.asarray(volume["size_xyz"], dtype=np.float32),
            "source": sax_nifti_path,
        }

    origin, direction, inv_direction = _build_local_frame_from_plane_records(records)
    return {
        "origin": origin,
        "direction": direction,
        "inv_direction": inv_direction,
        "spacing": np.array([1.0, 1.0, 1.0], dtype=np.float32),
        "size_xyz": None,
        "source": None,
    }


def _resolve_plane_dataset_image(record, path):
    candidates = []
    raw_path = record.get("mask_png")
    if isinstance(raw_path, str) and raw_path:
        candidates.append(raw_path)
        candidates.append(os.path.join(path, "mask_slices_png", os.path.basename(raw_path)))

    slice_num_one_based = record.get("slice_number_one_based")
    if slice_num_one_based is not None:
        candidates.append(os.path.join(path, "mask_slices_png", f"slice_{int(slice_num_one_based):02d}.png"))
    slice_idx_zero_based = record.get("slice_index_zero_based")
    if slice_idx_zero_based is not None:
        candidates.append(os.path.join(path, "mask_slices_png", f"slice_{int(slice_idx_zero_based) + 1:02d}.png"))

    for candidate in candidates:
        if candidate and os.path.exists(candidate):
            return candidate
    raise FileNotFoundError(f"Could not resolve mask PNG for record: {record}")


def _read_plane_position_dataset(path, tetra):
    plane_path = os.path.join(path, "plane_positions.json")
    with open(plane_path, "r", encoding="utf-8-sig") as f:
        records = json.load(f)

    if not isinstance(records, list) or not records:
        raise ValueError(f"Expected non-empty list in {plane_path}")

    frame = _resolve_plane_dataset_frame(path, records)
    local_vertices = world_to_local_sax(tetra.vertices, frame["origin"], frame["inv_direction"])
    tetra = BasicTetrahedra(vertices=local_vertices, cells=tetra.cells, colors=tetra.colors)

    bounds_min = local_vertices.min(axis=0).astype(np.float32)
    bounds_max = local_vertices.max(axis=0).astype(np.float32)

    plane_local_coords = []
    train_cam_infos = []
    test_cam_infos = []

    for record in records:
        image_path = _resolve_plane_dataset_image(record, path)
        gt_img = Image.open(image_path).convert("RGB")
        plane_origin_world = np.asarray(record["plane_origin_world"], dtype=np.float32)[None, :]
        plane_local = world_to_local_sax(plane_origin_world, frame["origin"], frame["inv_direction"])[0]
        plane_local_coords.append(plane_local)

        slice_idx = int(record.get("slice_index_zero_based", len(train_cam_infos) + len(test_cam_infos)))
        image_name = os.path.basename(image_path)
        camera = _build_camera(
            slice_idx,
            np.array([0.0, 0.0, float(plane_local[2])], dtype=np.float32),
            gt_img,
            "z",
            image_name,
            bounds_min,
            bounds_max,
        )
        if slice_idx > -1 and slice_idx < 250 and slice_idx % 20 == 7:
            test_cam_infos.append(camera)
        else:
            train_cam_infos.append(camera)

    if plane_local_coords:
        plane_local_coords = np.asarray(plane_local_coords, dtype=np.float32)
        bounds_min[2] = min(bounds_min[2], float(plane_local_coords[:, 2].min()))
        bounds_max[2] = max(bounds_max[2], float(plane_local_coords[:, 2].max()))
        updated_train = []
        updated_test = []
        for camera in train_cam_infos:
            updated_train.append(
                camera._replace(
                    slice_bounds_min=bounds_min.copy(),
                    slice_bounds_max=bounds_max.copy(),
                )
            )
        for camera in test_cam_infos:
            updated_test.append(
                camera._replace(
                    slice_bounds_min=bounds_min.copy(),
                    slice_bounds_max=bounds_max.copy(),
                )
            )
        train_cam_infos = updated_train
        test_cam_infos = updated_test

    return tetra, train_cam_infos, test_cam_infos


def _read_camera_info_dataset(path, tetra, image_name_prefix=None):
    camera_info_path = os.path.join(path, "camera_info.json")
    with open(camera_info_path, "r", encoding="utf-8-sig") as f:
        camera_info = json.load(f)

    records = camera_info.get("cameras", [])
    if not isinstance(records, list) or not records:
        raise ValueError(f"Expected non-empty 'cameras' list in {camera_info_path}")

    bounds_min = np.asarray(
        camera_info.get("slice_bounds_min", tetra.vertices.min(axis=0)),
        dtype=np.float32,
    )
    bounds_max = np.asarray(
        camera_info.get("slice_bounds_max", tetra.vertices.max(axis=0)),
        dtype=np.float32,
    )
    axis_name = str(camera_info.get("axis", "z")).lower()
    split_mode = str(camera_info.get("split", "train_only")).lower()

    train_cam_infos = []
    test_cam_infos = []
    for idx, record in enumerate(records):
        image_name = record.get("file_name")
        if not image_name:
            raise ValueError(f"Camera record {idx} in {camera_info_path} is missing 'file_name'")
        image_path = os.path.join(path, "mask_slices_png", image_name)
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Missing mask slice image {image_path}")

        gt_img = Image.open(image_path).convert("RGB")

        if "camera_position_xyz" in record:
            cam_pos = np.asarray(record["camera_position_xyz"], dtype=np.float32)
        elif "plane_coordinate_local_z" in record:
            cam_pos = np.array([0.0, 0.0, float(record["plane_coordinate_local_z"])], dtype=np.float32)
        else:
            raise ValueError(
                f"Camera record for {image_name} must contain 'camera_position_xyz' or 'plane_coordinate_local_z'"
            )
        plane_origin_world = None
        plane_normal_world = None
        plane_x_axis_world = None
        plane_y_axis_world = None
        plane_x_step = None
        plane_y_step = None
        if _has_explicit_plane_geometry(record):
            plane_origin_world = np.asarray(record["plane_origin_world_xyz"], dtype=np.float32)
            plane_normal_world = np.asarray(record["plane_normal_world_xyz"], dtype=np.float32)
            plane_x_axis_world = np.asarray(record["plane_x_axis_world_xyz"], dtype=np.float32)
            plane_y_axis_world = np.asarray(record["plane_y_axis_world_xyz"], dtype=np.float32)
            plane_x_step = float(record.get("plane_x_step", 1.0))
            plane_y_step = float(record.get("plane_y_step", 1.0))

        source_idx = int(record.get("source_slice_index", record.get("slice_index_zero_based", idx)))
        image_name_for_camera = image_name
        if image_name_prefix:
            image_name_for_camera = f"{image_name_prefix}_{image_name}"
        camera = _build_camera(
            source_idx,
            cam_pos,
            gt_img,
            axis_name,
            image_name_for_camera,
            bounds_min,
            bounds_max,
            plane_origin_world=plane_origin_world,
            plane_normal_world=plane_normal_world,
            plane_x_axis_world=plane_x_axis_world,
            plane_y_axis_world=plane_y_axis_world,
            plane_x_step=plane_x_step,
            plane_y_step=plane_y_step,
        )
        if split_mode == "train_only":
            train_cam_infos.append(camera)
        elif source_idx > -1 and source_idx < 250 and source_idx % 20 == 7:
            test_cam_infos.append(camera)
        else:
            train_cam_infos.append(camera)

    return tetra, train_cam_infos, test_cam_infos


def _read_multiaxis_camera_dataset(path, tetra):
    axis_names = _detect_multiaxis_camera_dataset(path)
    if not axis_names:
        raise ValueError(f"No multiaxis camera bundle found under {path}")

    def _round_robin_merge(axis_to_cams):
        merged = []
        positions = {axis_name: 0 for axis_name in axis_names}
        while True:
            took_any = False
            for axis_name in axis_names:
                cams = axis_to_cams.get(axis_name, [])
                pos = positions[axis_name]
                if pos < len(cams):
                    merged.append(cams[pos])
                    positions[axis_name] += 1
                    took_any = True
            if not took_any:
                break
        return merged

    def _sequential_merge(axis_to_cams):
        merged = []
        for axis_name in axis_names:
            merged.extend(axis_to_cams.get(axis_name, []))
        return merged

    axis_train_map = {}
    axis_test_map = {}

    for axis_name in axis_names:
        axis_path = os.path.join(path, axis_name)
        _, axis_train, axis_test = _read_camera_info_dataset(
            axis_path,
            tetra,
            image_name_prefix=axis_name,
        )
        axis_train_map[axis_name] = axis_train
        axis_test_map[axis_name] = axis_test

    if MULTIAXIS_MERGE_MODE == "sequential":
        train_cam_infos = _sequential_merge(axis_train_map)
        test_cam_infos = _sequential_merge(axis_test_map)
    else:
        train_cam_infos = _round_robin_merge(axis_train_map)
        test_cam_infos = _round_robin_merge(axis_test_map)

    next_uid = 0
    train_cam_infos = [cam._replace(uid=idx) for idx, cam in enumerate(train_cam_infos)]
    next_uid = len(train_cam_infos)
    test_cam_infos = [cam._replace(uid=next_uid + idx) for idx, cam in enumerate(test_cam_infos)]

    return tetra, train_cam_infos, test_cam_infos


def _read_png_axis_dataset(path, tetra, device, scale, extension):
    bounds_min = tetra.vertices.min(axis=0)
    bounds_max = tetra.vertices.max(axis=0)
    train_cam_infos = []
    test_cam_infos = []
    sax_info_path = os.path.join(path, "sax_info.json")
    sax_info = None
    if os.path.exists(sax_info_path):
        with open(sax_info_path, "r", encoding="utf-8-sig") as f:
            sax_info = json.load(f)
    if sax_info is not None and sax_info.get("mesh_storage") == "raw_world":
        origin = np.asarray(sax_info["origin"], dtype=np.float32)
        direction = np.asarray(sax_info["direction"], dtype=np.float32).reshape(3, 3)
        inv_direction = np.linalg.inv(direction)
        tetra = BasicTetrahedra(
            vertices=world_to_local_sax(tetra.vertices, origin, inv_direction),
            cells=tetra.cells,
            colors=tetra.colors,
        )
        bounds_min = tetra.vertices.min(axis=0)
        bounds_max = tetra.vertices.max(axis=0)

    axes = ["z"] if sax_info is not None else ["y"]
    for axis in axes:
        image_size = get_axis_image_size(path, axis)
        if image_size is None:
            continue

        axis_dir = os.path.join(path, axis)
        image_files = sorted(
            [f for f in os.listdir(axis_dir) if f.endswith(extension)],
            key=lambda f: int(f.split("_")[-1].split(".")[0]),
        )
        slice_records = None
        if sax_info is not None and isinstance(sax_info.get("slice_records"), list):
            slice_records = [r for r in sax_info["slice_records"] if str(r.get("file_name", "")).endswith(extension)]
            if slice_records:
                image_files = [r["file_name"] for r in slice_records]
        if not image_files:
            continue

        num_slices = len(image_files)
        for local_idx, img_file in enumerate(image_files):
            if sax_info is not None:
                slice_bounds_min = np.asarray(sax_info["slice_bounds_min"], dtype=np.float32)
                slice_bounds_max = np.asarray(sax_info["slice_bounds_max"], dtype=np.float32)
                idx = int(img_file.split("_")[-1].split(".")[0])
                if slice_records is not None:
                    record = next((r for r in slice_records if r["file_name"] == img_file), None)
                    if record is None:
                        continue
                    source_idx = int(record.get("source_slice_index", idx))
                    if "plane_origin_world_xyz" in record and sax_info is not None:
                        origin = np.asarray(sax_info["origin"], dtype=np.float32)
                        direction = np.asarray(sax_info["direction"], dtype=np.float32).reshape(3, 3)
                        inv_direction = np.linalg.inv(direction)
                        plane_origin_world = np.asarray(record["plane_origin_world_xyz"], dtype=np.float32)[None, :]
                        cam_pos = world_to_local_sax(plane_origin_world, origin, inv_direction)[0]
                    elif "camera_position_xyz" in record:
                        cam_pos = np.asarray(record["camera_position_xyz"], dtype=np.float32)
                    elif "plane_coordinate_local_z" in record:
                        cam_pos = np.array([0.0, 0.0, float(record["plane_coordinate_local_z"])], dtype=np.float32)
                    else:
                        cam_pos = np.array([0.0, 0.0, idx * float(sax_info["spacing"][2])], dtype=np.float32)
                else:
                    source_idx = idx
                    cam_pos = np.array([0.0, 0.0, idx * float(sax_info["spacing"][2])], dtype=np.float32)
            else:
                idx = int(img_file.split("_")[-1].split(".")[0])
                source_idx = idx
                slice_bounds_min = bounds_min
                slice_bounds_max = bounds_max
                try:
                    cam_pos = torch.from_numpy(
                        map_slice_index_to_bounds(idx, num_slices, axis, bounds_min, bounds_max)
                    ).to(device).cpu().numpy()
                except Exception:
                    cam_pos = _fallback_camera_position(idx, axis, num_slices, scale).to(device).cpu().numpy()

            img_path = os.path.join(axis_dir, img_file)
            try:
                gt_img = Image.open(img_path).convert("RGB")
            except Exception as exc:
                print(f"Error processing {axis}/{img_file}: {exc}")
                continue

            camera = _build_camera(
                source_idx,
                cam_pos,
                gt_img,
                axis,
                img_file,
                slice_bounds_min,
                slice_bounds_max,
            )
            split_idx = source_idx if sax_info is not None else idx
            if split_idx > -1 and split_idx < 250 and split_idx % 20 == 7:
                test_cam_infos.append(camera)
            elif split_idx > -1 and split_idx < 250:
                train_cam_infos.append(camera)

    return tetra, train_cam_infos, test_cam_infos


def _read_nifti_sax_dataset(path, tetra):
    nifti_path = resolve_nifti_path(path)
    if nifti_path is None:
        return None

    volume = load_nifti_volume(nifti_path)
    local_vertices = world_to_local_sax(
        tetra.vertices,
        volume["origin"],
        volume["inv_direction"],
    )
    local_tetra = BasicTetrahedra(vertices=local_vertices, cells=tetra.cells, colors=tetra.colors)

    sax_array = volume["array"]
    spacing = volume["spacing"]
    size_xyz = np.asarray(volume["size_xyz"], dtype=np.float32)
    slice_bounds_min = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    slice_bounds_max = np.array(
        [
            (size_xyz[0] - 1.0) * spacing[0],
            (size_xyz[1] - 1.0) * spacing[1],
            (size_xyz[2] - 1.0) * spacing[2],
        ],
        dtype=np.float32,
    )
    train_cam_infos = []
    test_cam_infos = []

    for slice_idx in range(sax_array.shape[0]):
        plane_z = float(slice_idx) * float(spacing[2])
        cam_pos = np.array([0.0, 0.0, plane_z], dtype=np.float32)
        gt_img = normalize_slice_to_rgb(sax_array[slice_idx])
        camera = _build_camera(
            slice_idx,
            cam_pos,
            gt_img,
            "z",
            f"sax_{slice_idx:03d}.png",
            slice_bounds_min,
            slice_bounds_max,
        )
        if slice_idx > -1 and slice_idx % 20 == 7:
            test_cam_infos.append(camera)
        else:
            train_cam_infos.append(camera)

    return nifti_path, local_tetra, train_cam_infos, test_cam_infos


def readMRISceneInfo(path, white_background, eval, init_mesh, extension=".png"):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    scale = [1.0, 1.0, 1.0]

    tetra_path = _resolve_tetra_path(path)
    use_camera_info_dataset = _detect_camera_info_dataset(path)
    multiaxis_camera_axes = _detect_multiaxis_camera_dataset(path)
    use_plane_dataset = _detect_plane_dataset(path)
    use_png_dataset = os.path.exists(os.path.join(path, "sax_info.json")) and os.path.isdir(os.path.join(path, "z"))

    tetra_path, tetra = _load_or_create_tetrahedra(path, tetra_path, scale)
    if use_camera_info_dataset:
        tetra, train_cam_infos, test_cam_infos = _read_camera_info_dataset(path, tetra)
    elif multiaxis_camera_axes:
        tetra, train_cam_infos, test_cam_infos = _read_multiaxis_camera_dataset(path, tetra)
    elif use_plane_dataset:
        tetra, train_cam_infos, test_cam_infos = _read_plane_position_dataset(path, tetra)
    elif use_png_dataset:
        tetra, train_cam_infos, test_cam_infos = _read_png_axis_dataset(path, tetra, device, scale, extension)
    else:
        nifti_result = _read_nifti_sax_dataset(path, tetra)
        if nifti_result is not None:
            _, tetra, train_cam_infos, test_cam_infos = nifti_result
        else:
            tetra, train_cam_infos, test_cam_infos = _read_png_axis_dataset(path, tetra, device, scale, extension)

    nerf_normalization = getNerfppNorm(train_cam_infos)
    ply_path, pcd = _load_or_create_point_cloud(path, tetra.vertices)

    return SceneInfo(
        point_cloud=pcd,
        tetrahedra=tetra,
        train_cameras=train_cam_infos,
        test_cameras=test_cam_infos,
        nerf_normalization=nerf_normalization,
        ply_path=ply_path,
    )


sceneLoadTypeCallbacks = {
    "MRI": readMRISceneInfo,
}
