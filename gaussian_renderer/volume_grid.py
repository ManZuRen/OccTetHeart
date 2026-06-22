from __future__ import annotations

from typing import Optional, Sequence, Tuple

import torch


def _parse_grid_origin_xyz(
    grid_origin_xyz: Optional[Sequence[float]],
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if grid_origin_xyz is None:
        return torch.zeros((3,), device=device, dtype=dtype)
    return torch.as_tensor(grid_origin_xyz, device=device, dtype=dtype)


def _parse_spacing_xyz(
    voxel_spacing_xyz: Sequence[float],
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    spacing = torch.as_tensor(voxel_spacing_xyz, device=device, dtype=dtype)
    if spacing.numel() != 3:
        raise ValueError("voxel_spacing_xyz must have exactly 3 values in x/y/z order.")
    return torch.clamp_min(spacing, 1e-8)


def _parse_block_size_zyx(block_size: Sequence[int]) -> Tuple[int, int, int]:
    if isinstance(block_size, int):
        value = int(block_size)
        if value <= 0:
            raise ValueError("block_size must be positive.")
        return value, value, value
    if len(block_size) != 3:
        raise ValueError("block_size must be an int or a 3-tuple in z/y/x order.")
    block_size_zyx = (int(block_size[0]), int(block_size[1]), int(block_size[2]))
    if min(block_size_zyx) <= 0:
        raise ValueError("block_size values must all be positive.")
    return block_size_zyx


def _volume_shape_to_size_xyz(volume_shape_zyx: Sequence[int], device: torch.device) -> torch.Tensor:
    if len(volume_shape_zyx) != 3:
        raise ValueError("volume_shape_zyx must be a length-3 sequence in z/y/x order.")
    z, y, x = [int(v) for v in volume_shape_zyx]
    if min(z, y, x) <= 0:
        raise ValueError(f"Invalid volume shape {tuple(volume_shape_zyx)}.")
    return torch.tensor([x, y, z], device=device, dtype=torch.long)


def binary_dice_loss_3d(
    input_tensor: torch.Tensor,
    target_tensor: torch.Tensor,
    weight: Optional[torch.Tensor] = None,
    smooth: float = 1.0,
) -> torch.Tensor:
    n = target_tensor.shape[0]
    input_flat = input_tensor.reshape(n, -1)
    target_flat = target_tensor.reshape(n, -1)
    intersection = input_flat * target_flat
    dice_eff = (2.0 * intersection.sum(1) + smooth) / (
        input_flat.sum(1) + target_flat.sum(1) + smooth + 1e-6
    )
    loss = (1.0 - dice_eff) * weight if weight is not None else (1.0 - dice_eff)
    return loss.mean()
