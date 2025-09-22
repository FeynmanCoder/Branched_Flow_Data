# branched_flow/simulation.py (請使用此最終版本)

import time
import numpy as np
from typing import Dict, Any
from tqdm import tqdm

from core.backends import get_backend
from core.field import FieldData
from analysis.statistics import StatisticsManager
from analysis.io import HistoryManager, ExportManager
from analysis.visualization import VisualizationManager
from analysis.snapshot import assemble_snapshot_from_slices
from physics.updates import update_rule_registry
from physics.potentials import fourier_synthesis_gpu
import timer

class Simulation:
    def __init__(self, params: Dict[str, Any], initial_field: FieldData, initial_particles: Dict[str, np.ndarray]):
        self.params = params
        self.is_quiet = self.params.get('quiet_mode', False)
        
        self.y0_all = initial_particles['y0']
        self.p0_all = initial_particles['p0']
        self.w0_all = initial_particles['w0']
        
        if not self.is_quiet: print("\n--- 初始化核心組件 (僅一次) ---")
        self.backend = get_backend(self.params, initial_field)
        self.xp = self.backend.xp
        
        self.backend.setup_computation(len(self.y0_all))
        
        self.force_field = self.xp.asarray(initial_field.get_field_matrix(), dtype=self.backend.dtype)
        
        self.stats_manager = StatisticsManager(self.params, self.xp)
        self.history_manager = HistoryManager(self.params)
        self.export_manager = ExportManager(self.params, self.xp)
        self.viz_manager = VisualizationManager(self.params, initial_field, self.xp)

        self.timer = timer

    def set_timer(self, timer):
        """從外部接收計時器物件。"""
        self.timer = timer

    def _convert_potential_to_force_gpu(self, potential_matrix_gpu):
        """在 GPU 上將勢能轉換為力場。"""
        dy = self.params['dy_potential']
        fy = self.xp.zeros_like(potential_matrix_gpu)
        fy[1:-1, :] = - (potential_matrix_gpu[2:, :] - potential_matrix_gpu[:-2, :]) / (2 * dy)
        fy[0, :] = - (potential_matrix_gpu[1, :] - potential_matrix_gpu[0, :]) / dy
        fy[-1, :] = - (potential_matrix_gpu[-1, :] - potential_matrix_gpu[-2, :]) / dy
        return fy

    def run_generation_cycle_on_gpu(self, seed: int):
        """【高效能版本】在 GPU 上執行一次完整的數據生成週期。"""
        p = self.params
        
        alpha = p['potential_alpha']
        amplitude = p['potential_amplitude']

        ny, nx = self.force_field.shape
        potential_gpu = fourier_synthesis_gpu(
            xp=self.xp, ny=ny, nx=nx, dx=p['dx_potential'], dy=p['dy_potential'], 
            alpha=alpha,
            amplitude=amplitude,
            seed=seed
        )

        self.force_field = self._convert_potential_to_force_gpu(potential_gpu)
        
        _, y_traj, p_traj = self.backend.run_single_batch(
            self.force_field, self.y0_all, self.p0_all, self.w0_all
        )
        
        snapshot_data = {
            "batch_num": 1, 
            "field_before_batch": self.force_field, 
            "y_reshaped": y_traj,
            "weights": self.w0_all.copy(), 
            "stats": {}
        }
        self.viz_manager.plot_batch_report(snapshot_data)
        
        
        
    def reset_for_new_run(self, seed: int):
        """重設模擬狀態以開始一次新的獨立演化。"""
        p = self.params
        
        # 根據新的種子在GPU上生成一個全新的隨機勢能場
        ny, nx = self.force_field.shape
        potential_gpu = fourier_synthesis_gpu(
            xp=self.xp, ny=ny, nx=nx, dx=p['dx_potential'], dy=p['dy_potential'], 
            alpha=p['potential_alpha'],
            amplitude=p['potential_amplitude'],
            seed=seed
        )
        # 將其轉換為力場並更新到 self.force_field
        self.force_field = self._convert_potential_to_force_gpu(potential_gpu)
        print(f"  [重設] 已為第 {seed} 組生成新的隨機種子力場。")
        
            
    def run(self, num_batches: int = 0):
        """用於 main.py 的原始、詳細的模擬方法。"""
        p = self.params
        # 如果沒有從外部傳入批次數，則使用設定檔中的預設值
        total_batches = num_batches if num_batches > 0 else p['num_batches']
        
        update_rule_func = update_rule_registry[p['potential_update_rule']]
        
        # 使用tqdm顯示內部批次迴圈的進度
        progress_bar = tqdm(range(1, total_batches + 1), desc=f"  組 {p.get('simulation_id', 0)} 批次進度", leave=False)
        
        for batch_num in progress_bar:
            field_before_update = self.force_field.copy() # 複製一份，避免後續修改影響
            
            with self.timer.record("物理計算 (run_single_batch)"):
                total_modification, y_traj, p_traj = self.backend.run_single_batch(
                    field_before_update, self.y0_all, self.p0_all, self.w0_all
                )

            # 在AI生成模式下，我們需要為每個批次都產生快照
            should_snapshot = True if p.get('enable_ai_data_export', False) else self.history_manager.should_snapshot(batch_num, total_batches)

            if should_snapshot:
                stats = {}
                snapshot_data = {
                    "batch_num": batch_num, "field_before_batch": field_before_update,
                    "y_reshaped": y_traj, "p_reshaped": p_traj,
                    "weights": self.w0_all, "stats": stats
                }
                
                # --- 精準測量存檔 (包括GPU分析和IO) 時間 ---
                with self.timer.record("資料處理與存檔 (plot_batch_report)"):
                    self.viz_manager.plot_batch_report(snapshot_data)
            
            # 力場更新通常很快，暫不計時
            update_rule_func(self.force_field, total_modification, p)
            
    def analyze_and_visualize(self):
        """用於 main.py 的原始分析與可視化方法。"""
        if self.is_quiet: return
        
        history = self.history_manager.get_history()
        
        self.viz_manager.plot_batch_reports(history)
        self.viz_manager.plot_summary_stats(history)

        print("\n--- 最終束流分析 (從狀態切片組裝) ---")
        snapshot_data = assemble_snapshot_from_slices(self.snapshot_slices)
        
        if snapshot_data["x"].size > 0:
            self.viz_manager.plot_beam_snapshot(snapshot_data)
        else:
            print("警告：未能組裝任何粒子，跳過束流圖繪製。")

        self.viz_manager.finalize()