# branched_flow/core/cpu_backend.py

import numpy as np
from typing import Dict, Any, Tuple
from scipy.interpolate import RegularGridInterpolator

from .base import Backend  # <--- 从 base.py 导入，打破循环
from physics import interactions, boundaries

class CPUBackend(Backend):
    """使用 NumPy 和 SciPy 在 CPU 上执行计算的后端。"""
    def _setup_backend_specifics(self):
        self.xp = np
        self.dtype = np.float64 if self.params.get('precision', 'float64') == 'float64' else np.float32

    def setup_computation(self, num_particles: int):
        print("  [Backend Setup] Initializing CPU backend components...")
        p = self.params
        self.interaction_func = interactions.interaction_model_registry_cpu[p['interaction_model']]
        self.interaction_params = tuple(p['interaction_params'].values())
        self.boundary_func = boundaries.boundary_condition_registry_cpu[p['boundary_condition']]
        self.x_values = np.arange(p['x0'], p['x_end'], p['dx'])
        
        self.y_traj = self.xp.zeros((num_particles, len(self.x_values)), dtype=self.dtype)
        self.p_traj = self.xp.zeros((num_particles, len(self.x_values)), dtype=self.dtype)
        self.weights = self.xp.zeros(num_particles, dtype=self.dtype)

    def run_single_batch(self, force_field_cpu, y0, p0, w0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        self.y_traj[:, 0] = y0
        self.p_traj[:, 0] = p0
        self.weights[:] = w0

        y_traj_out, p_traj_out = self._rk4_integrator_cpu(force_field_cpu)
        total_modification = self._modify_field_cpu(force_field_cpu, y_traj_out)

        return total_modification, y_traj_out, p_traj_out

    def get_snapshot_slice(self, y_traj_cpu, p_traj_cpu, x_target, x_width) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        x_min = x_target - x_width / 2.0
        x_max = x_target + x_width / 2.0
        
        col_indices = np.where((self.x_values >= x_min) & (self.x_values <= x_max))[0]
        
        if len(col_indices) == 0:
            return np.array([]), np.array([]), np.array([])
            
        y_slice = y_traj_cpu[:, col_indices].flatten()
        p_slice = p_traj_cpu[:, col_indices].flatten()
        x_slice = np.tile(self.x_values[col_indices], y_traj_cpu.shape[0]).flatten()
        
        return x_slice, y_slice, p_slice

    def _rk4_integrator_cpu(self, force_field):
        p = self.params
        m, v_x, y_end = p['m'], p['v_x'], p['y_end']
        interpolator = RegularGridInterpolator(
            (self.field_data_obj.get_y_grid_coords(), self.field_data_obj.get_x_grid_coords()),
            force_field, bounds_error=False, fill_value=0
        )
        for i in range(len(self.x_values) - 1):
            dt = (self.x_values[i+1] - self.x_values[i]) / v_x
            for j in range(self.y_traj.shape[0]):
                y, p_y = self.y_traj[j, i], self.p_traj[j, i]
                if np.isnan(y):
                    self.y_traj[j, i+1], self.p_traj[j, i+1] = y, p_y
                    continue
                
                x_i = self.x_values[i]
                k1_y = p_y / m
                k1_p = interpolator((y, x_i))
                k2_y = (p_y + 0.5 * dt * k1_p) / m
                k2_p = interpolator((y + 0.5 * dt * k1_y, x_i + 0.5*dt*v_x))
                k3_y = (p_y + 0.5 * dt * k2_p) / m
                k3_p = interpolator((y + 0.5 * dt * k2_y, x_i + 0.5*dt*v_x))
                k4_y = (p_y + dt * k3_p) / m
                k4_p = interpolator((y + dt * k3_y, self.x_values[i+1]))
                
                y_new = y + (dt / 6.0) * (k1_y + 2*k2_y + 2*k3_y + k4_y)
                p_y_new = p_y + (dt / 6.0) * (k1_p + 2*k2_p + 2*k3_p + k4_p)

                self.y_traj[j, i+1], self.p_traj[j, i+1] = self.boundary_func(y_new, p_y_new, 0.0, y_end)
        return self.y_traj, self.p_traj

    def _modify_field_cpu(self, force_field, y_traj):
        p = self.params
        y_end = p['y_end']
        cutoff_sq = p.get('interaction_cutoff_radius', 5.0)**2
        
        total_mod = np.zeros_like(force_field)
        ny, nx = total_mod.shape
        grid_x = self.field_data_obj.get_x_grid_coords()
        grid_y = self.field_data_obj.get_y_grid_coords()
        
        interpolator = RegularGridInterpolator((grid_y, grid_x), force_field, bounds_error=False, fill_value=0)

        downsample = p.get('mod_downsample', 1)
        num_x_steps_ds = y_traj[:, ::downsample].shape[1]
        traj_x = np.tile(self.x_values[::downsample], y_traj.shape[0])
        traj_y = y_traj[:, ::downsample].flatten()
        
        traj_y_periodic = traj_y % y_end
        valid_mask = ~np.isnan(traj_y_periodic)
        
        vals_at_points = np.full(len(traj_x), np.nan)
        points_to_interp = np.column_stack((traj_y_periodic[valid_mask], traj_x[valid_mask]))
        vals_at_points[valid_mask] = interpolator(points_to_interp)

        expanded_weights = np.repeat(self.weights, num_x_steps_ds)

        for gy_idx in range(ny):
            for gx_idx in range(nx):
                gx, gy = grid_x[gx_idx], grid_y[gy_idx]
                force_at_grid = force_field[gy_idx, gx_idx]
                
                dx_sq = (traj_x - gx)**2
                dy_raw = traj_y - gy
                dy_mic = dy_raw - y_end * np.round(dy_raw / y_end)
                dist_sq = dx_sq + dy_mic**2
                
                nearby = np.where((dist_sq < cutoff_sq) & valid_mask)[0]
                
                mod_sum = 0.0
                for idx in nearby:
                    dist = np.sqrt(dist_sq[idx])
                    force_at_particle = vals_at_points[idx]
                    mod = self.interaction_func(dist, force_at_particle, force_at_grid, *self.interaction_params)
                    mod_sum += mod * expanded_weights[idx]
                
                total_mod[gy_idx, gx_idx] = mod_sum
        return total_mod
