# branched_flow/core/integrators.py

import math
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from numba import cuda

from .field import FieldData
from ..physics import boundaries

# ================= GPU Kernels =================

@cuda.jit(device=True)
def gpu_interpolate_bilinear(y, x, field_matrix, x_min, y_min, dx, dy):
    ny, nx = field_matrix.shape
    x_idx_float = (x - x_min) / dx
    y_idx_float = (y - y_min) / dy
    x1 = int(math.floor(x_idx_float))
    y1 = int(math.floor(y_idx_float))
    if x1 < 0 or x1 >= nx - 1 or y1 < 0 or y1 >= ny - 1: return 0.0
    x2, y2 = x1 + 1, y1 + 1
    Q11, Q12 = field_matrix[y1, x1], field_matrix[y2, x1]
    Q21, Q22 = field_matrix[y1, x2], field_matrix[y2, x2]
    x_rem, y_rem = x_idx_float - x1, y_idx_float - y1
    f_xy1 = (1.0 - y_rem) * Q11 + y_rem * Q12
    f_xy2 = (1.0 - y_rem) * Q21 + y_rem * Q22
    return (1.0 - x_rem) * f_xy1 + x_rem * f_xy2

def create_gpu_integrator_kernel():
    boundary_funcs = boundaries.boundary_condition_registry_gpu
    @cuda.jit
    def rk4_integrator_kernel(y_traj, p_traj, x_vals, m, v_x, y_end, field, x_min, y_min, dx, dy, bc_mode):
        p_idx = cuda.grid(1)
        if p_idx >= y_traj.shape[0]: return
        y, p_y = y_traj[p_idx, 0], p_traj[p_idx, 0]
        
        for i in range(len(x_vals) - 1):
            if math.isnan(y):
                y_traj[p_idx, i + 1], p_traj[p_idx, i + 1] = y, p_y
                continue
            
            x_i = x_vals[i]
            dt = (x_vals[i+1] - x_i) / v_x
            
            k1_y = p_y / m
            k1_p = gpu_interpolate_bilinear(y, x_i, field, x_min, y_min, dx, dy)
            k2_y = (p_y + 0.5 * dt * k1_p) / m
            k2_p = gpu_interpolate_bilinear(y + 0.5 * dt * k1_y, x_i + 0.5*dt*v_x, field, x_min, y_min, dx, dy)
            k3_y = (p_y + 0.5 * dt * k2_p) / m
            k3_p = gpu_interpolate_bilinear(y + 0.5 * dt * k2_y, x_i + 0.5*dt*v_x, field, x_min, y_min, dx, dy)
            k4_y = (p_y + dt * k3_p) / m
            k4_p = gpu_interpolate_bilinear(y + dt * k3_y, x_vals[i+1], field, x_min, y_min, dx, dy)
            
            y += (dt / 6.0) * (k1_y + 2*k2_y + 2*k3_y + k4_y)
            p_y += (dt / 6.0) * (k1_p + 2*k2_p + 2*k3_p + k4_p)
            
            if bc_mode == 0: y, p_y = boundary_funcs['periodic'](y, p_y, 0.0, y_end)
            elif bc_mode == 1: y, p_y = boundary_funcs['reflecting'](y, p_y, 0.0, y_end)
            elif bc_mode == 2: y, p_y = boundary_funcs['kill'](y, p_y, 0.0, y_end)
            
            y_traj[p_idx, i + 1], p_traj[p_idx, i + 1] = y, p_y
    return rk4_integrator_kernel

# ================= CPU Integrator =================

def rk4_integrator_cpu(p, field_data, y_traj, p_traj, x_vals, boundary_func):
    """CPU 版本的 RK4 积分器。"""
    num_particles = y_traj.shape[0]
    m, v_x, y_end = p['m'], p['v_x'], p['y_end']

    force_interpolator = RegularGridInterpolator(
        (field_data.get_y_grid_coords(), field_data.get_x_grid_coords()),
        field_data.get_field_matrix(), bounds_error=False, fill_value=0
    )

    for i in range(len(x_vals) - 1):
        dt = (x_vals[i+1] - x_vals[i]) / v_x
        for j in range(num_particles):
            y, p_y = y_traj[j, i], p_traj[j, i]
            if np.isnan(y):
                y_traj[j, i+1], p_traj[j, i+1] = y, p_y
                continue
            
            x_i = x_vals[i]
            k1_y = p_y / m
            k1_p = force_interpolator((y, x_i))
            k2_y = (p_y + 0.5 * dt * k1_p) / m
            k2_p = force_interpolator((y + 0.5 * dt * k1_y, x_i + 0.5*dt*v_x))
            k3_y = (p_y + 0.5 * dt * k2_p) / m
            k3_p = force_interpolator((y + 0.5 * dt * k2_y, x_i + 0.5*dt*v_x))
            k4_y = (p_y + dt * k3_p) / m
            k4_p = force_interpolator((y + dt * k3_y, x_vals[i+1]))
            
            y_new = y + (dt / 6.0) * (k1_y + 2*k2_y + 2*k3_y + k4_y)
            p_y_new = p_y + (dt / 6.0) * (k1_p + 2*k2_p + 2*k3_p + k4_p)

            y_traj[j, i+1], p_traj[j, i+1] = boundary_func(y_new, p_y_new, 0.0, y_end)
            
    return y_traj, p_traj
