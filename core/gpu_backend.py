# branched_flow/core/gpu_backend.py

import numpy as np
import cupy as cp
from typing import Dict, Any, Tuple

from .base import Backend
from physics import interactions
from . import gpu_kernels
from .gpu_kernels import create_modify_field_kernel_optimized

class GPUBackend(Backend):
    """使用 CuPy 和 Numba 在 GPU 上执行计算的后端。"""
    def _setup_backend_specifics(self):
        self.xp = cp
        self.dtype = self.xp.float32 if self.params.get('precision', 'float32') == 'float32' else self.xp.float64
    
    def setup_computation(self, num_particles: int):
        print("  [Backend Setup] Compiling CUDA kernels for GPU...")
        p = self.params
        
        # 【新增】定義互動模型的名稱到整數標誌的映射
        self.interaction_mode_map = {
            'force_dependent_exp_decay': 0,
            'gaussian': 1
        }
        
        # 獲取當前選擇的互動模型參數
        self.interaction_params = tuple(p['interaction_params'].values())
        
        x_values = np.arange(p['x0'], p['x_end'], p['dx'])
        self.x_values_gpu = self.xp.asarray(x_values, dtype=self.dtype)
        
        self.y_traj_gpu = self.xp.zeros((num_particles, len(x_values)), dtype=self.dtype)
        self.p_traj_gpu = self.xp.zeros((num_particles, len(x_values)), dtype=self.dtype)
        self.weights_gpu = self.xp.zeros(num_particles, dtype=self.dtype)
        
        self.boundary_mode_map = {'periodic': 0, 'reflecting': 1, 'kill': 2}
        self._compile_kernels()
        

    def _compile_kernels(self):
        """編譯所有需要的CUDA内核。"""
        self.rk4_integrator_kernel = gpu_kernels.create_rk4_kernel()
        
        # 編譯並使用優化版的核心
        self.modify_field_kernel_optimized = create_modify_field_kernel_optimized()

        self.get_field_val_kernel = gpu_kernels.get_field_val_kernel
        self.collect_slice_kernel = gpu_kernels.collect_slice_kernel
    
    def run_single_batch(self, force_field_gpu, y0, p0, w0) -> Tuple[Any, Any, Any]:
        """執行一個完整批次的模擬計算。"""
        p = self.params
        
        # 1-4 步完全不變...
        num_particles = self.y_traj_gpu.shape[0]
        self.y_traj_gpu[:, 0] = self.xp.asarray(y0, dtype=self.dtype)
        self.p_traj_gpu[:, 0] = self.xp.asarray(p0, dtype=self.dtype)
        self.weights_gpu[:] = self.xp.asarray(w0, dtype=self.dtype)
        threads = 128
        blocks = (num_particles + threads - 1) // threads
        bc_mode = self.boundary_mode_map[p['boundary_condition']]
        self.rk4_integrator_kernel[blocks, threads](
            self.y_traj_gpu, self.p_traj_gpu, self.x_values_gpu, p['m'], p['v_x'], p['y_end'],
            force_field_gpu, self.field_data_obj.x_min, self.field_data_obj.y_min,
            self.field_data_obj.dx, self.field_data_obj.dy, bc_mode
        )
        downsample = p.get('mod_downsample', 1)
        num_x_ds = self.y_traj_gpu[:, ::downsample].shape[1]
        y_flat_ds = self.y_traj_gpu[:, ::downsample].flatten()
        x_tiled_ds = self.xp.tile(self.x_values_gpu[::downsample], num_particles)
        traj_points_ds = self.xp.column_stack((x_tiled_ds, y_flat_ds))
        vals_at_points = self.xp.zeros(traj_points_ds.shape[0], dtype=self.dtype)
        threads = 256
        blocks = (traj_points_ds.shape[0] + threads - 1) // threads
        self.get_field_val_kernel[blocks, threads](
            vals_at_points, traj_points_ds, force_field_gpu, self.field_data_obj.x_min,
            self.field_data_obj.y_min, self.field_data_obj.dx, self.field_data_obj.dy, p['y_end']
        )
        
        # 5. 【核心修改】呼叫優化版的力場修正核心，並傳遞整數標誌
        total_mod = self.xp.zeros_like(force_field_gpu)
        grid_x_gpu = self.xp.asarray(self.field_data_obj.get_x_grid_coords(), dtype=self.dtype)
        grid_y_gpu = self.xp.asarray(self.field_data_obj.get_y_grid_coords(), dtype=self.dtype)
        
        threads_per_block = 256
        blocks_per_grid = (traj_points_ds.shape[0] + threads_per_block - 1) // threads_per_block
        
        # 獲取當前互動模型的整數標誌
        interaction_mode = self.interaction_mode_map[p['interaction_model']]
        
        self.modify_field_kernel_optimized[blocks_per_grid, threads_per_block](
            total_mod, vals_at_points, traj_points_ds, grid_y_gpu, grid_x_gpu,
            interaction_mode,          # <-- 傳遞整數標誌
            self.interaction_params,   # <-- 傳遞純粹的參數元組
            p['y_end'], p['interaction_cutoff_radius'],
            force_field_gpu, self.field_data_obj.y_min, self.field_data_obj.x_min,
            self.field_data_obj.dy, self.field_data_obj.dx, 
            self.weights_gpu, num_x_ds
        )
        
        self.xp.cuda.runtime.deviceSynchronize()
        
        # 6. 返回計算結果
        return total_mod, self.y_traj_gpu, self.p_traj_gpu
    
    def get_snapshot_slice(self, y_traj_gpu, p_traj_gpu, x_target, x_width) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """从GPU上的轨迹数据中收集一个窗口内的所有点。"""
        p = self.params
        num_p, num_x = y_traj_gpu.shape
        num_x_in_width = int(np.ceil(x_width / p['dx'])) + 2
        max_points = num_p * num_x_in_width

        x_out = self.xp.zeros(max_points, dtype=self.dtype)
        y_out = self.xp.zeros(max_points, dtype=self.dtype)
        p_out = self.xp.zeros(max_points, dtype=self.dtype)
        count = self.xp.zeros(1, dtype=self.xp.int32)

        threads = 256
        grid = (num_p * num_x + threads - 1) // threads
        
        self.collect_slice_kernel[grid, threads](y_traj_gpu, p_traj_gpu, x_out, y_out, p_out, count, self.x_values_gpu, x_target, x_width)
        self.xp.cuda.runtime.deviceSynchronize()
        
        num_collected = int(count.get()[0])
        
        return self.xp.asnumpy(x_out[:num_collected]), self.xp.asnumpy(y_out[:num_collected]), self.xp.asnumpy(p_out[:num_collected])
