import os
from typing import Optional, Tuple

import nibabel as nib
import numpy as np
from PIL import Image

try:
    import SimpleITK as sitk
except ImportError:
    sitk = None


AXIS_MAPPING = {
    "x": 0,
    "y": 1,
    "z": 2,
    "axial": 2,
    "sagittal": 0,
    "coronal": 1,
}


def get_axis_index(axis_name: str) -> int:
    return AXIS_MAPPING.get(axis_name.lower(), 2)


def get_axis_image_size(data_root: str, axis: str) -> Optional[Tuple[int, int]]:
    axis_path = os.path.join(data_root, axis)
    if not os.path.exists(axis_path):
        return None

    image_files = [f for f in os.listdir(axis_path) if f.endswith(".png")]
    if not image_files:
        return None

    image = Image.open(os.path.join(axis_path, image_files[0]))
    width, height = image.size
    return (height, width)


def map_slice_index_to_bounds(
    slice_index: int,
    total_slices: int,
    axis_name: str,
    bounds_min: np.ndarray,
    bounds_max: np.ndarray,
) -> np.ndarray:
    axis_idx = get_axis_index(axis_name)
    coord = np.zeros(3, dtype=np.float32)

    if total_slices <= 1:
        coord[axis_idx] = 0.5 * (bounds_min[axis_idx] + bounds_max[axis_idx])
        return coord

    ratio = float(slice_index) / float(total_slices - 1)
    coord[axis_idx] = bounds_min[axis_idx] + ratio * (bounds_max[axis_idx] - bounds_min[axis_idx])
    return coord


def resolve_nifti_path(data_root: str) -> Optional[str]:
    candidates = [
        os.path.join(data_root, "frame01.nii.gz"),
        os.path.join(data_root, "sax.nii.gz"),
        os.path.join(data_root, "sax", "frame01.nii.gz"),
        os.path.join(data_root, "sax", "sax.nii.gz"),
    ]
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    return None


def load_nifti_volume(nifti_path: str):
    if sitk is not None:
        image = sitk.ReadImage(nifti_path)
        array = sitk.GetArrayFromImage(image).astype(np.float32)  # [z, y, x]
        spacing = np.array(image.GetSpacing(), dtype=np.float32)
        origin = np.array(image.GetOrigin(), dtype=np.float32)
        direction = np.array(image.GetDirection(), dtype=np.float32).reshape(3, 3)
        size_xyz = np.array(image.GetSize(), dtype=np.int32)
    else:
        image = nib.load(nifti_path)
        data = np.asarray(image.get_fdata(dtype=np.float32), dtype=np.float32)  # [x, y, z]
        array = np.transpose(data, (2, 1, 0))  # [z, y, x]
        affine = np.asarray(image.affine, dtype=np.float32)
        basis = affine[:3, :3]
        spacing = np.linalg.norm(basis, axis=0).astype(np.float32)
        safe_spacing = np.where(spacing > 0, spacing, 1.0).astype(np.float32)
        direction = (basis / safe_spacing[None, :]).astype(np.float32)
        origin = affine[:3, 3].astype(np.float32)
        size_xyz = np.array(data.shape, dtype=np.int32)
    inv_direction = np.linalg.inv(direction)
    return {
        "image": image,
        "array": array,
        "spacing": spacing,
        "origin": origin,
        "direction": direction,
        "inv_direction": inv_direction,
        "size_xyz": size_xyz,
    }


def world_to_local_sax(points_world: np.ndarray, origin: np.ndarray, inv_direction: np.ndarray) -> np.ndarray:
    return np.dot(points_world - origin[None, :], inv_direction.T).astype(np.float32)


def normalize_slice_to_rgb(slice_array: np.ndarray) -> Image.Image:
    slice_array = np.asarray(slice_array, dtype=np.float32)
    min_value = float(slice_array.min())
    max_value = float(slice_array.max())
    if max_value <= min_value:
        normalized = np.zeros(slice_array.shape, dtype=np.uint8)
    else:
        scaled = (slice_array - min_value) / (max_value - min_value)
        normalized = np.clip(np.round(scaled * 255.0), 0, 255).astype(np.uint8)
    rgb = np.stack([normalized, normalized, normalized], axis=-1)
    return Image.fromarray(rgb, mode="RGB")
