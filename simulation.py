# branched_flow/simulation.py

import numpy as np
from typing import Dict, Any
from tqdm import tqdm

from core.backends import get_backend
from core.field import FieldData
from analysis.io import save_ai_training_data
from physics.potentials import fourier_synthesis_gpu

class Simulation:
    def __init__(self, params: Dict[str, Any], initial_field: FieldData, initial_particles: Dict[str, np.ndarray]):
        self.params = params
        self.is_quiet = self.params.get('quiet_mode', True) # 在生成模式下，預設為靜默
        
        self.y0_all = initial_particles['y0']
        self.p0_all = initial_particles['p0']
        self.w0_all = initial_particles['w0']
        
        if not self.is_quiet: print("\n--- 初始化核心組件 ---")
        self.backend = get_backend(self.params, initial_field)
        self.xp = self.backend.xp
        
        self.backend.setup_computation(len(self.y0_all))
        
        # 初始力場僅用於確定形狀和類型
        self.force_field = self.xp.asarray(initial_field.get_field_matrix(), dtype=self.backend.dtype)
        
        self.timer = None

    def set_timer(self, timer_instance):
        """從外部接收計時器物件。"""
        self.timer = timer_instance

    def _convert_potential_to_force_gpu(self, potential_matrix_gpu):
        """在 GPU 上將勢能轉換為力場。"""
        dy = self.params['dy_potential']
        fy = self.xp.zeros_like(potential_matrix_gpu)
        fy[1:-1, :] = - (potential_matrix_gpu[2:, :] - potential_matrix_gpu[:-2, :]) / (2 * dy)
        fy[0, :] = - (potential_matrix_gpu[1, :] - potential_matrix_gpu[0, :]) / dy
        fy[-1, :] = - (potential_matrix_gpu[-1, :] - potential_matrix_gpu[-2, :]) / dy
        return fy

    def reset_for_new_run(self, seed: int):
        """
        重設模擬狀態以開始一次新的獨立演化。
        這在資料生成模式中至關重要，確保每組資料都是從一個新的隨機勢場開始。
        """
        p = self.params
        
        # 根據新的種子在GPU上生成一個全新的隨機勢能場
        ny, nx = self.force_field.shape
        with self.timer.record("生成隨機勢場 (GPU)"):
            potential_gpu = fourier_synthesis_gpu(
                xp=self.xp, ny=ny, nx=nx, dx=p['dx_potential'], dy=p['dy_potential'], 
                alpha=p['potential_alpha'],
                amplitude=p['potential_amplitude'],
                seed=seed
            )
        # 將其轉換為力場並更新到 self.force_field
        self.force_field = self._convert_potential_to_force_gpu(potential_gpu)
        if not self.is_quiet:
            print(f"  [重設] 已為第 {seed} 組生成新的隨機種子力場。")
            
    def run(self, num_batches: int = 1):
        """
        專為 AI 資料生成精簡的運行循環。
        此函數現在只負責核心計算和儲存結果，移除了所有不必要的分析和繪圖。
        """
        p = self.params
        
        progress_bar = tqdm(range(1, num_batches + 1), desc=f"  組 {p.get('simulation_id', 0)} 批次進度", leave=False, disable=self.is_quiet)
        
        for batch_num in progress_bar:
            # 核心物理計算
            with self.timer.record("物理計算 (run_single_batch)"):
                _, y_traj, _ = self.backend.run_single_batch(
                    self.force_field, self.y0_all, self.p0_all, self.w0_all
                )

            # 資料處理與儲存
            with self.timer.record("資料處理與存檔"):
                save_ai_training_data(
                    params=p,
                    batch_num=batch_num,
                    potential_field=self.force_field,
                    y_trajectory=y_traj,
                    weights=self.w0_all,
                    xp=self.xp
                )
