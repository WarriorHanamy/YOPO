"""GPU-accelerated ESDF computation via CuPy EDT.

Provides exact Euclidean Signed Distance Field computation on GPU,
with automatic fallback to scipy when CuPy is not installed.
"""

import numpy as np
import torch as th

_CUPY_AVAILABLE = False

try:
    import cupy as cp
    from cupyx.scipy.ndimage import distance_transform_edt as cupy_edt

    _CUPY_AVAILABLE = True
except ImportError:
    pass


def _compute_edt_scipy(seed_mask, voxel_size):
    """CPU EDT via scipy."""
    from scipy.ndimage import distance_transform_edt

    return distance_transform_edt(seed_mask) * voxel_size


def _compute_edt_cupy(seed_mask, voxel_size):
    """GPU EDT via CuPy — exact, fast."""
    seed_gpu = cp.asarray(seed_mask)
    dist_gpu = cupy_edt(seed_gpu)
    result = cp.asnumpy(dist_gpu) * voxel_size
    return result


def compute_esdf(occupancy, voxel_size, device):
    """
    Compute signed Euclidean distance field from occupancy grid.

    Uses CuPy (GPU, exact) when available, falls back to scipy.

    @param[in] occupancy   Binary occupancy grid (nx, ny, nz), 1=obstacle [uint8]
    @param[in] voxel_size  Physical size of one voxel [m]
    @param[in] device      Target torch device for output tensor
    @return Signed distance field (nx, ny, nz) on `device`, positive=free [float32, m]
    """
    obstacle_mask = occupancy.astype(bool)
    free_mask = ~obstacle_mask

    if not obstacle_mask.any():
        return th.full(occupancy.shape, 1e10, dtype=th.float32, device=device)
    if not free_mask.any():
        return th.full(occupancy.shape, -1e10, dtype=th.float32, device=device)

    edt_fn = _compute_edt_cupy if _CUPY_AVAILABLE else _compute_edt_scipy

    dist_to_obstacle = edt_fn(free_mask, voxel_size)
    dist_to_free = edt_fn(obstacle_mask, voxel_size)

    sdf = dist_to_obstacle.astype(np.float32)
    sdf[obstacle_mask] = -dist_to_free[obstacle_mask]

    return th.from_numpy(sdf).float().to(device)
