import glob
import os

import numpy as np
import open3d as o3d
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage import distance_transform_edt

from ..schema import config


class SafetyLoss(nn.Module):
    def __init__(self, L):
        super().__init__()
        self.traj_num = config.traj_num
        self.map_expand_min = np.array(config.map.map_expand_min)
        self.map_expand_max = np.array(config.map.map_expand_max)
        self.d0 = config.safety.d0
        self.r = config.safety.r

        self._L = L
        self.sgm_time = config.sgm_time
        self.eval_points = 30
        self.device = self._L.device
        self.time_integral = True

        self.voxel_size = 0.2
        self.min_bounds = None
        self.max_bounds = None
        self.sdf_shapes = None
        self.sdf_maps = None
        self._esdf_built = False

    def build(self):
        """Build ESDF maps from PLY files. Called lazily on first forward() pass."""
        if self._esdf_built:
            return
        data_dir = config.dataset.resolved
        print('Building ESDF map...')
        self.sdf_maps = self._get_sdf_from_ply(data_dir)
        print('Map built!')
        self._esdf_built = True

    def forward(self, Df, Dp, map_id):
        """
        Args:
            Dp: decision parameters: (batch_size, 3, 3) -> [px, vx, ax; py, vy, ay; pz, vz, az]
            Df: fixed parameters: (batch_size, 3, 3) -> [px, vx, ax; py, vy, ay; pz, vz, az]
            map_id: (batch_size) which esdf map to query
        Returns:
            cost_colli: (batch_size) -> safety loss
        """
        if not self._esdf_built:
            self.build()

        batch_size = Dp.shape[0]
        L = self._L.unsqueeze(0).expand(batch_size, -1, -1)
        coe = self._get_coefficient_from_derivative(Dp, Df, L)

        dt = self.sgm_time / self.eval_points
        t_list = th.linspace(dt, self.sgm_time, self.eval_points, device=self.device)
        t_list = t_list.view(1, -1, 1).expand(batch_size, -1, -1)

        pos_coe = self._get_position_from_coeff(coe, t_list)
        pos_batch = pos_coe.reshape(-1, self.traj_num * pos_coe.shape[1], 3)

        cost, dist = self._get_distance_cost(pos_batch, map_id)

        if self.time_integral:
            cost_colli = cost.reshape(-1, pos_coe.shape[1]).mean(dim=-1)
        else:
            vel_coe = self._get_velocity_from_coeff(coe, t_list)
            vel_coe = vel_coe.norm(dim=-1)
            line_integral_cost = (cost.reshape(-1, pos_coe.shape[1]) * vel_coe * dt).sum(dim=1)
            line_length = (vel_coe * dt).sum(dim=1)
            cost_colli = line_integral_cost / line_length

        return cost_colli

    def _get_distance_cost(self, pos, map_id):
        """
        pos:     (B, N, 3) - positions in world frame
        map_id:  (B) - which sdf_map to query per batch element
        NOTE: Direct self.sdf_maps.expand(B, -1, -1, -1, -1) is the most memory-efficient and fastest, but only supports a single map.
              Using self.sdf_maps[map_id] results in significant memory usage and latency due to data copying.
              As a compromise, we adopt a map-cropping (_get_batch_sdf) to support multiple maps.
        """
        B, N, _ = pos.shape

        sdf_maps, local_origin, local_shape = self._get_batch_sdf(pos, map_id)

        grid = (pos - local_origin.unsqueeze(1)) / self.voxel_size  # (B, N, 3)

        grid_point = (
            2.0 * grid / (local_shape - 1).unsqueeze(1) - 1.0
        )  # (B, N, 3), normalize to [-1, 1]

        grid_point = grid_point.view(B, 1, 1, N, 3)
        grid_point = th.clamp(grid_point, min=-0.99, max=0.99)

        dist_query = F.grid_sample(
            sdf_maps,
            grid_point,
            mode='bilinear',
            padding_mode='zeros',
            align_corners=True,
        )
        dist_query = dist_query.view(B, N)

        cost = self._cost_function(dist_query)
        return cost, dist_query

    def _cost_function(self, d):
        return th.exp(-(d - self.d0) / self.r)

    def _get_coefficient_from_derivative(self, Dp, Df, L):
        coefficient = th.zeros(Dp.shape[0], 18, device=self.device)

        for i in range(3):
            d = th.cat([Df[:, i, :], Dp[:, i, :]], dim=1).unsqueeze(-1)
            coe = (L @ d).squeeze()
            coefficient[:, 6 * i : 6 * (i + 1)] = coe

        return coefficient

    def _get_position_from_coeff(self, coe, t):
        t_power = th.stack([th.ones_like(t), t, t**2, t**3, t**4, t**5], dim=-1).squeeze(-2)

        coe_x = coe[:, 0:6]
        coe_y = coe[:, 6:12]
        coe_z = coe[:, 12:18]

        x = th.sum(t_power * coe_x.unsqueeze(1), dim=-1)
        y = th.sum(t_power * coe_y.unsqueeze(1), dim=-1)
        z = th.sum(t_power * coe_z.unsqueeze(1), dim=-1)

        pos = th.stack([x, y, z], dim=-1)
        return pos

    def _get_velocity_from_coeff(self, coe, t):
        t_power = th.stack([th.ones_like(t), 2 * t, 3 * t**2, 4 * t**3, 5 * t**4], dim=-1).squeeze(
            -2
        )

        coe_x = coe[:, 1:6]
        coe_y = coe[:, 7:12]
        coe_z = coe[:, 13:18]

        vx = th.sum(t_power * coe_x.unsqueeze(1), dim=-1)
        vy = th.sum(t_power * coe_y.unsqueeze(1), dim=-1)
        vz = th.sum(t_power * coe_z.unsqueeze(1), dim=-1)

        vel = th.stack([vx, vy, vz], dim=-1)
        return vel

    def _get_batch_sdf(self, pos, map_id):
        """
        Crop all maps with the corresponding map_id in the batch to the same size and cover the pos.
        """
        min_bounds = self.min_bounds[map_id]
        sdf_shapes = self.sdf_shapes[map_id]

        min_pos = pos.amin(dim=1)
        max_pos = pos.amax(dim=1)
        min_indices = ((min_pos - min_bounds) / self.voxel_size).int()
        max_indices = ((max_pos - min_bounds) / self.voxel_size).int()
        spans = max_indices - min_indices
        max_spans = spans.amax(dim=0)
        centers = (min_indices + max_indices) // 2
        min_indices = centers - max_spans // 2 - 5
        max_indices = centers + max_spans // 2 + 5

        new_min_indices = min_indices.clamp(min=0)
        underflow_amount = new_min_indices - min_indices
        min_indices = new_min_indices
        max_indices = max_indices + underflow_amount

        new_max_indices = th.minimum(max_indices, sdf_shapes.int())
        overflow_amount = max_indices - new_max_indices
        max_indices = new_max_indices
        min_indices = min_indices - overflow_amount

        if (min_indices < 0).any():
            min_underflow = th.minimum(min_indices, th.zeros_like(min_indices))
            shift = (-min_underflow).max(dim=0).values
            min_indices = min_indices + shift

        sdf_maps = th.stack(
            [
                self.sdf_maps[map_idx][
                    0,
                    :,
                    min_idx[2] : max_idx[2],
                    min_idx[1] : max_idx[1],
                    min_idx[0] : max_idx[0],
                ]
                for map_idx, min_idx, max_idx in zip(
                    map_id.tolist(), min_indices.tolist(), max_indices.tolist()
                )
            ]
        )
        local_origin = min_indices * self.voxel_size + min_bounds
        local_shape = max_indices - min_indices
        return sdf_maps, local_origin, local_shape

    def _get_sdf_from_ply(self, path):
        sorted_files = self._read_sorted_ply_files(path)
        sdf_maps = []
        min_bounds, max_bounds, sdf_shapes = [], [], []

        for file in sorted_files:
            pcd = o3d.io.read_point_cloud(file)
            min_bound = np.array(pcd.get_min_bound()) - self.map_expand_min
            max_bound = np.array(pcd.get_max_bound()) + self.map_expand_max
            points = np.asarray(pcd.points)
            print(
                f'    {os.path.basename(file)}: x=({min_bound[0] + self.map_expand_min[0]:.2f}, {max_bound[0] - self.map_expand_max[0]:.2f}), '
                f'y=({min_bound[1] + self.map_expand_min[1]:.2f}, {max_bound[1] - self.map_expand_max[1]:.2f}), '
                f'z=({min_bound[2] + self.map_expand_min[2]:.2f}, {max_bound[2] - self.map_expand_max[2]:.2f})'
            )

            sdf_shape = np.ceil((max_bound - min_bound) / self.voxel_size).astype(int)
            voxel_indices = ((points - min_bound) / self.voxel_size).astype(int)

            valid_mask = np.all((voxel_indices >= 0) & (voxel_indices < sdf_shape), axis=1)
            voxel_indices = voxel_indices[valid_mask]

            occupancy = np.zeros(sdf_shape, dtype=np.uint8)
            occupancy[tuple(voxel_indices.T)] = 1

            obstacle_mask = occupancy == 1
            free_mask = occupancy == 0

            dist_to_obstacle = distance_transform_edt(free_mask) * self.voxel_size
            dist_inside_obstacle = distance_transform_edt(obstacle_mask) * self.voxel_size

            dist_to_obstacle[obstacle_mask] = -dist_inside_obstacle[obstacle_mask]

            sdf_tensor = (
                th.from_numpy(dist_to_obstacle)
                .float()
                .unsqueeze(0)
                .unsqueeze(0)
                .permute(0, 1, 4, 3, 2)
                .to(self.device)
            )

            sdf_maps.append(sdf_tensor)
            sdf_shapes.append(sdf_tensor.shape[-3:][::-1])
            min_bounds.append(min_bound)
            max_bounds.append(max_bound)

        self.min_bounds = th.tensor(np.array(min_bounds), device=self.device).float()
        self.max_bounds = th.tensor(np.array(max_bounds), device=self.device).float()
        self.sdf_shapes = th.tensor(np.array(sdf_shapes), device=self.device).float()
        return sdf_maps

    def _read_sorted_ply_files(self, path):
        ply_files = glob.glob(os.path.join(path, 'pointcloud-*.ply'))

        def extract_index(filename):
            base = os.path.basename(filename)
            number_part = base.replace('pointcloud-', '').replace('.ply', '')
            return int(number_part)

        sorted_ply_files = sorted(ply_files, key=extract_index)

        return sorted_ply_files
