# branched_flow/core/gpu_kernels.py

import math
from numba import cuda
from physics import boundaries
from physics import interactions 

# ==============================================================================
#                      GPU Device Functions and Kernels
# ==============================================================================

@cuda.jit(device=True)
def gpu_interpolate_bilinear(y, x, field, x_min, y_min, dx, dy):
    """在GPU上进行双线性插值。"""
    ny, nx = field.shape
    x_idx_float = (x - x_min) / dx
    y_idx_float = (y - y_min) / dy
    x1 = int(math.floor(x_idx_float))
    y1 = int(math.floor(y_idx_float))
    if not (0 <= x1 < nx - 1 and 0 <= y1 < ny - 1):
        return 0.0
    x2, y2 = x1 + 1, y1 + 1
    rem_x, rem_y = x_idx_float - x1, y_idx_float - y1
    q11, q12, q21, q22 = field[y1, x1], field[y2, x1], field[y1, x2], field[y2, x2]
    f_xy1 = (1.0 - rem_y) * q11 + rem_y * q12
    f_xy2 = (1.0 - rem_y) * q21 + rem_y * q22
    return (1.0 - rem_x) * f_xy1 + rem_x * f_xy2

def create_rk4_kernel():
    """工厂函数，创建并返回RK4积分器内核。"""
    periodic_bc = boundaries.boundary_condition_registry_gpu['periodic']
    reflecting_bc = boundaries.boundary_condition_registry_gpu['reflecting']
    kill_bc = boundaries.boundary_condition_registry_gpu['kill']

    @cuda.jit
    def rk4_kernel(y_traj, p_traj, x_vals, m, v_x, y_end, field, x_min, y_min, dx, dy, bc_mode):
        p_idx = cuda.grid(1)
        if p_idx >= y_traj.shape[0]:
            return
        
        y, p_y = y_traj[p_idx, 0], p_traj[p_idx, 0]
        
        for i in range(len(x_vals) - 1):
            if math.isnan(y):
                y_traj[p_idx, i + 1], p_traj[p_idx, i + 1] = y, p_y
                continue
            
            x_i = x_vals[i]
            dt = (x_vals[i+1] - x_i) / v_x
            
            k1_y = p_y / m
            k1_p = gpu_interpolate_bilinear(y, x_i, field, x_min, y_min, dx, dy)
            k2_y = (p_y + 0.5*dt*k1_p) / m
            k2_p = gpu_interpolate_bilinear(y + 0.5*dt*k1_y, x_i + 0.5*dt*v_x, field, x_min, y_min, dx, dy)
            k3_y = (p_y + 0.5*dt*k2_p) / m
            k3_p = gpu_interpolate_bilinear(y + 0.5*dt*k2_y, x_i + 0.5*dt*v_x, field, x_min, y_min, dx, dy)
            k4_y = (p_y + dt*k3_p) / m
            k4_p = gpu_interpolate_bilinear(y + dt*k3_y, x_vals[i+1], field, x_min, y_min, dx, dy)
            
            y += (dt/6.0) * (k1_y + 2*k2_y + 2*k3_y + k4_y)
            p_y += (dt/6.0) * (k1_p + 2*k2_p + 2*k3_p + k4_p)
            
            if bc_mode == 0:
                y, p_y = periodic_bc(y, p_y, 0.0, y_end)
            elif bc_mode == 1:
                y, p_y = reflecting_bc(y, p_y, 0.0, y_end)
            elif bc_mode == 2:
                y, p_y = kill_bc(y, p_y, 0.0, y_end)
            
            y_traj[p_idx, i + 1], p_traj[p_idx, i + 1] = y, p_y
            
    return rk4_kernel

def create_modify_field_kernel(interaction_func):
    """工厂函数，为特定的相互作用模型创建力场修改内核。"""
    @cuda.jit
    def modify_field_kernel(total_mod, vals_at_points, traj_points, grid_x, grid_y, i_params, y_end, cutoff_radius, field, x_min, y_min, dx, dy, weights, num_x_steps_ds):
        grid_idx_start = cuda.grid(1)
        stride = cuda.gridsize(1)
        cutoff_sq = cutoff_radius * cutoff_radius
        
        for grid_idx in range(grid_idx_start, total_mod.size, stride):
            gy_idx = grid_idx // total_mod.shape[1]
            gx_idx = grid_idx % total_mod.shape[1]
            gx, gy = grid_x[gx_idx], grid_y[gy_idx]
            
            mod_sum = 0.0
            for traj_idx in range(traj_points.shape[0]):
                px, py = traj_points[traj_idx, 0], traj_points[traj_idx, 1]
                if math.isnan(py):
                    continue
                
                dy_mic = py - gy
                dy_mic -= y_end * round(dy_mic / y_end)
                dist_sq = (px - gx)**2 + dy_mic**2
                
                if dist_sq < cutoff_sq:
                    dist = math.sqrt(dist_sq)
                    p_idx = traj_idx // num_x_steps_ds
                    mod = interaction_func(dist, vals_at_points[traj_idx], field[gy_idx, gx_idx], *i_params)
                    mod_sum += mod * weights[p_idx]
            
            total_mod[gy_idx, gx_idx] = mod_sum
            
    return modify_field_kernel

@cuda.jit
def get_field_val_kernel(vals_out, traj_points, field, x_min, y_min, dx, dy, y_end):
    """获取轨迹点上力场值的内核。"""
    idx = cuda.grid(1)
    if idx >= traj_points.shape[0]:
        return
    x, y = traj_points[idx, 0], traj_points[idx, 1]
    y_periodic = y % y_end
    vals_out[idx] = gpu_interpolate_bilinear(y_periodic, x, field, x_min, y_min, dx, dy)

@cuda.jit
def collect_slice_kernel(y_traj, p_traj, x_out, y_out, p_out, count, x_vals, x_target, x_width):
    """收集一个窗口内所有轨迹点的内核。"""
    start_idx = cuda.grid(1)
    stride = cuda.gridsize(1)
    num_p, num_x = y_traj.shape
    total_points = num_p * num_x
    max_out = x_out.shape[0]
    x_min, x_max = x_target - x_width / 2.0, x_target + x_width / 2.0

    for i in range(start_idx, total_points, stride):
        p_idx, x_idx = i // num_x, i % num_x
        curr_x = x_vals[x_idx]
        if x_min <= curr_x <= x_max:
            write_pos = cuda.atomic.add(count, 0, 1)
            if write_pos < max_out:
                x_out[write_pos] = curr_x
                y_out[write_pos] = y_traj[p_idx, x_idx]
                p_out[write_pos] = p_traj[p_idx, x_idx]



def create_modify_field_kernel_optimized():
    """
    【最終修正版】工廠函式，創建使用原子散射演算法的力場修改核心。
    使用 interaction_mode 整數標誌來選擇物理模型，以規避 Numba 的限制。
    """
    # 為了在核心函式中使用，我們需要提前獲取這些 device function
    force_dep_decay_logic = interactions.interaction_model_registry_gpu['force_dependent_exp_decay']
    gaussian_logic = interactions.interaction_model_registry_gpu['gaussian']

    @cuda.jit
    def modify_field_kernel_optimized(total_mod, vals_at_points, traj_points, grid_y_coords, grid_x_coords, 
                                      interaction_mode, i_params, # <-- 修改點：接收 mode 和 params
                                      y_end, cutoff_radius, field, y_min, x_min, dy, dx, weights, num_x_steps_ds):
        # 讓每個執行緒負責一個軌跡點
        traj_idx = cuda.grid(1)
        if traj_idx >= traj_points.shape[0]:
            return

        px, py = traj_points[traj_idx, 0], traj_points[traj_idx, 1]
        
        if math.isnan(py):
            return

        # --- 計算此軌跡點影響的網格範圍 (這部分邏輯不變) ---
        px_idx = (px - x_min) / dx
        py_idx = (py - y_min) / dy
        radius_in_grid_units_x = int(cutoff_radius / dx) + 1
        radius_in_grid_units_y = int(cutoff_radius / dy) + 1
        ny, nx = total_mod.shape
        start_gx_idx = max(0, int(px_idx - radius_in_grid_units_x))
        end_gx_idx = min(nx, int(px_idx + radius_in_grid_units_x))
        start_gy_idx = max(0, int(py_idx - radius_in_grid_units_y))
        end_gy_idx = min(ny, int(py_idx + radius_in_grid_units_y))

        for gy_idx in range(start_gy_idx, end_gy_idx):
            for gx_idx in range(start_gx_idx, end_gx_idx):
                gx, gy = grid_x_coords[gx_idx], grid_y_coords[gy_idx]
                dy_mic = py - gy
                dy_mic -= y_end * round(dy_mic / y_end)
                dist_sq = (px - gx)**2 + dy_mic**2
                
                if dist_sq < cutoff_radius**2:
                    dist = math.sqrt(dist_sq)
                    p_idx = traj_idx // num_x_steps_ds
                    mod = 0.0
                    
                    # --- 【核心修改】根據 mode 標誌選擇要執行的物理邏輯 ---
                    if interaction_mode == 0: # 模式 0: force_dependent_exp_decay
                        # 手動解包參數
                        mod_sign, mod_a, mod_b = i_params
                        mod = force_dep_decay_logic(dist, vals_at_points[traj_idx], field[gy_idx, gx_idx], mod_sign, mod_a, mod_b)
                    elif interaction_mode == 1: # 模式 1: gaussian
                        # 手動解包參數
                        mod_sign, amplitude, sigma = i_params
                        mod = gaussian_logic(dist, vals_at_points[traj_idx], field[gy_idx, gx_idx], mod_sign, amplitude, sigma)

                    # 使用原子操作，安全地將影響值累加到網格點上
                    cuda.atomic.add(total_mod, (gy_idx, gx_idx), mod * weights[p_idx])

    return modify_field_kernel_optimized