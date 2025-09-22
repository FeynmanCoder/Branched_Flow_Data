# branched_flow/analysis/visualization.py

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import matplotlib.gridspec as gridspec
from typing import Dict, Any, List
import os
from PIL import Image  # <--- 確保 PIL/Pillow 已匯入
from matplotlib.collections import LineCollection
import sys

class VisualizationManager:
    """負責所有與可視化相關的編排工作。"""
    def __init__(self, params: Dict[str, Any], field_data_obj, xp):
        self.params = params
        self.field_data_obj = field_data_obj
        self.xp = xp
        self.is_live_plotting = params.get('live_plotting', False)
        self.output_path = self.params.get('plot_output_path', '.')
        self.fig = None
        self._setup_plotting()

    def _save_ai_data_pair(self, field_data, density_data, group_id, batch_num, base_path, train_or_test):
        """
        【高效能优化版】
        将力场和密度矩阵标准化后，直接保存为快速、无损的 .npy 格式。
        """
        print(f"  - 正在儲存: 力場 shape={field_data.shape}, 軌跡圖 shape={density_data.shape}")
        try:
            # 路径保持不变，只是现在里面储存的是 .npy 文件
            path_A = os.path.join(base_path, f"{train_or_test}A") # A folder for force fields
            path_B = os.path.join(base_path, f"{train_or_test}B") # B folder for density maps
            os.makedirs(path_A, exist_ok=True)
            os.makedirs(path_B, exist_ok=True)

            # 档名，只是后缀变为 .npy
            base_filename = f"group_{group_id:04d}_batch_{batch_num:003d}"
            filepath_A = os.path.join(path_A, f"{base_filename}.npy")
            filepath_B = os.path.join(path_B, f"{base_filename}.npy")

            # --- 1. 标准化并储存力场矩阵 ---
            field_min, field_max = np.min(field_data), np.max(field_data)
            if field_max - field_min > 1e-9:
                field_norm = (field_data - field_min) / (field_max - field_min)
            else:
                field_norm = np.zeros_like(field_data)
            
            # 使用 np.save，速度极快。指定数据类型为 float32 以节省空间
            np.save(filepath_A, field_norm.astype(np.float32))
            
            # --- 2. 标准化并储存粒子密度矩阵 ---
            if self.params.get('use_log_scale_plots', True):
                density_data = np.log1p(density_data)
            
            density_min, density_max = np.min(density_data), np.max(density_data)
            if density_max - density_min > 1e-9:
                density_norm = (density_data - density_min) / (density_max - density_min)
            else:
                density_norm = np.zeros_like(density_data)
                
            np.save(filepath_B, density_norm.astype(np.float32))

        except Exception as e:
            print(f"警告：保存 AI 数据对 (.npy) 时出错: {e}")

    def _setup_plotting(self):
        try:
            plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei'] 
            plt.rcParams['axes.unicode_minus'] = False
        except Exception:
            print("警告：設置中文字體失敗，部分標籤可能無法正確顯示。")
        
        os.makedirs(self.output_path, exist_ok=True)
        
        if self.is_live_plotting:
            self.fig = plt.figure(figsize=(22, 18))
            plt.ion()

    def plot_batch_reports(self, history: List[Dict[str, Any]]):
        if self.is_live_plotting or not history: return
        
        # 如果啟用 AI 數據導出，則跳過生成批次報告圖的提示
        if not self.params.get('enable_ai_data_export', False):
            print("\n開始生成各批次的綜合報告圖...")
            
        for snapshot in history:
            self.plot_batch_report(snapshot)

    # branched_flow/analysis/visualization.py -> plot_batch_report 函式


    def plot_batch_report(self, snapshot_data: Dict[str, Any]):
        # --- AI 資料匯出模式 ---
        if self.params.get('enable_ai_data_export', False):
            field_cpu = self.xp.asnumpy(snapshot_data['field_before_batch'])
            y_traj_gpu = snapshot_data['y_reshaped']
            
            x_vals_gpu = self.xp.arange(self.params['x0'], self.params['x_end'], self.params['dx'])
            x_flat_gpu = self.xp.tile(x_vals_gpu, y_traj_gpu.shape[0])
            y_flat_gpu = y_traj_gpu.flatten()
            mask_gpu = self.xp.isfinite(y_flat_gpu)
            
            hist_range = [
                [self.params['x0'], self.params['x_end']], 
                [0, self.params['y_end']]
            ]

            hist_gpu, _, _ = self.xp.histogram2d(
                x=x_flat_gpu[mask_gpu], 
                y=y_flat_gpu[mask_gpu], 
                bins=[self.params['density_bins_x'], self.params['density_bins_y']],
                range=hist_range
            )
            
            # 【最終修正】將 histogram 的 (nx, ny) 輸出轉置為 (ny, nx) 以匹配力場的慣例
            density_map_cpu = self.xp.asnumpy(hist_gpu.T)

            # ... (後續的儲存邏輯完全不變) ...
            base_path = self.params['ai_data_output_path']
            train_or_test = self.params.get('dataset_split', 'train')
            group_id = self.params.get('simulation_id', 0)
            batch_num = snapshot_data['batch_num']
            
            self._save_ai_data_pair(
                field_cpu, 
                density_map_cpu, 
                group_id, 
                batch_num, 
                base_path, 
                train_or_test
            )
            
            return 
        
        # --- 原有的詳細繪圖邏輯（僅在非AI的'simulate'模式下執行） ---
        p = self.params
        fig = plt.figure(figsize=(22, 18))
        self._draw_comprehensive_plot(fig, snapshot_data)

        file_path = os.path.join(self.output_path, f"report_batch_{snapshot_data['batch_num']:05d}.png")
        fig.savefig(file_path, dpi=100, bbox_inches='tight')
        plt.close(fig)

        if self.is_live_plotting:
            plt.draw()
            plt.pause(0.01)

    def plot_summary_stats(self, history: List[Dict[str, Any]]):
        stats_history = [s['stats'] for s in history if 'stats' in s]
        if not stats_history: return

        print("\n正在生成系統演化統計圖...")
        batch_numbers = [s['batch_num'] for s in history if 'stats' in s]
        
        fig, axes = plt.subplots(2, 2, figsize=(18, 10), sharex=True)
        fig.suptitle('系統演化統計圖', fontsize=16)
        ax1, ax2, ax3, ax4 = axes.flatten()

        ke = [s['total_kinetic_energy'] for s in stats_history]
        ax1.plot(batch_numbers, ke, 'o-', label='粒子末端總動能 $E_k$')
        ax1.set_ylabel('總動能 (Joule)'); ax1.set_title('動能演化'); ax1.legend(); ax1.grid(True)
        
        mean_f = [s['mean_field_val'] for s in stats_history]
        std_f = [s['std_field_val'] for s in stats_history]
        ax2.plot(batch_numbers, mean_f, 's-', color='red', label='平均力場 $\\bar{F_y}$')
        ax2.fill_between(batch_numbers, np.array(mean_f) - np.array(std_f), np.array(mean_f) + np.array(std_f), color='red', alpha=0.2, label='力場標準差 $\\sigma_{F_y}$')
        ax2.set_ylabel('力 (N)'); ax2.set_title('力場統計演化'); ax2.legend(); ax2.grid(True)

        foc_len = [s.get('focusing_length', np.nan) for s in stats_history]
        ax3.plot(batch_numbers, foc_len, 'd-', color='green', label='匯聚長度 $x_{foc}$')
        ax3.set_xlabel('批次數'); ax3.set_ylabel('x 座標 (m)'); ax3.set_title('匯聚長度演化'); ax3.legend(); ax3.grid(True)
        
        corr_len = [s.get('correlation_length', np.nan) for s in stats_history]
        ax4.plot(batch_numbers, corr_len, '^-', color='purple', label='相干長度 $L_c$')
        ax4.set_xlabel('批次數'); ax4.set_ylabel('長度 (m)'); ax4.set_title('力場相干長度演化'); ax4.legend(); ax4.grid(True)
        
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        file_path = os.path.join(self.output_path, "summary_statistics.png")
        fig.savefig(file_path, dpi=150, bbox_inches='tight')
        plt.close(fig)

    def plot_beam_snapshot(self, snapshot_data: Dict[str, np.ndarray]):
        print("\n正在生成最終束流快照圖...")
        fig = plt.figure(figsize=(24, 10))
        self._draw_beam_snapshot_plot(fig, snapshot_data)
        file_path = os.path.join(self.output_path, "beam_snapshot.png")
        fig.savefig(file_path, dpi=150, bbox_inches='tight')
        plt.close(fig)

    def finalize(self):
        if self.is_live_plotting: plt.ioff()
        
        if not self.params.get('enable_ai_data_export', False):
            print("\n所有繪圖已生成。")
            if not self.is_live_plotting:
                 print(f"所有圖像已保存到 '{self.output_path}' 文件夾。")
            else:
                plt.show()

    def _draw_comprehensive_plot(self, fig, snapshot_data):
        p = self.params
        fig.suptitle(f'第 {snapshot_data["batch_num"]} 批次粒子模拟综合报告', fontsize=28, y=0.97)
        gs = gridspec.GridSpec(2, 2, figure=fig, wspace=0.3, hspace=0.35)
        
        field = self.xp.asnumpy(snapshot_data['field_before_batch'])
        y_traj = self.xp.asnumpy(snapshot_data['y_reshaped'])
        p_traj = self.xp.asnumpy(snapshot_data['p_reshaped'])
        weights = snapshot_data.get('weights', np.ones(y_traj.shape[0]))

        # Panel 1: Field with Colorbar
        ax_field = fig.add_subplot(gs[0, 0])
        field_extent = [p['x0'], p['x_end'], 0.0, p['y_end']]
        im_field = ax_field.imshow(field, cmap='coolwarm', interpolation='antialiased',
                                   origin='lower', extent=field_extent, aspect='auto')
        ax_field.set_title('背景力场 (Fy)'); ax_field.set_xlabel('x (m)'); ax_field.set_ylabel('y (m)')
        ax_field.set_aspect('equal', adjustable='box')
        cbar_field = fig.colorbar(im_field, ax=ax_field, orientation='vertical')
        cbar_field.set_label('力大小 (Fy)')

        # Panel 2: Trajectories
        ax_traj = fig.add_subplot(gs[0, 1])
        im_traj_bg = ax_traj.imshow(field, cmap='coolwarm', interpolation='antialiased',
                                    origin='lower', extent=field_extent, aspect='auto', alpha=0.8)
        x_vals = np.arange(p['x0'], p['x_end'], p['dx'])
        for i in range(y_traj.shape[0]):
            jumps = np.where(np.abs(np.diff(y_traj[i, :])) > p['y_end'] / 2)[0] + 1
            for xs, ys in zip(np.split(x_vals, jumps), np.split(y_traj[i, :], jumps)):
                ax_traj.plot(xs, ys, alpha=0.6, linewidth=0.8)
        ax_traj.set_xlim(p['x0'], p['x_end']); ax_traj.set_ylim(0, p['y_end'])
        ax_traj.set_title('粒子轨迹与背景力场'); ax_traj.set_xlabel('x (m)'); ax_traj.set_ylabel('y (m)')
        ax_traj.set_aspect('equal', adjustable='box')
        cbar_traj = fig.colorbar(im_traj_bg, ax=ax_traj, orientation='vertical')
        cbar_traj.set_label('力大小 (Fy)')

        # Panel 3: Phase Space
        ax_phase_outer = fig.add_subplot(gs[1, 0])
        ax_phase_outer.set_title('不同时刻的 (y, py) 相空间分布'); ax_phase_outer.axis('off')
        phase_gs = gridspec.GridSpecFromSubplotSpec(3, 3, subplot_spec=gs[1, 0], hspace=0.7, wspace=0.3)
        time_indices = np.linspace(0, y_traj.shape[1] - 1, 9, dtype=int)
        for i, time_idx in enumerate(time_indices):
            ax = fig.add_subplot(phase_gs[i])
            ax.scatter(y_traj[:, time_idx], p_traj[:, time_idx], s=10, alpha=0.6, edgecolors='none')
            ax.set_title(f't = {time_idx * p["dx"] / p["v_x"]:.2f}s', fontsize=10)
            ax.set_xlim(0, p['y_end'])
            if i % 3 == 0: ax.set_ylabel('动量 $p_y$')
            if i // 3 == 2: ax.set_xlabel('位置 y')

        # Panel 4: Density with Colorbar
        ax_density = fig.add_subplot(gs[1, 1])
        use_log_scale = p.get('use_log_scale_plots', True)
        norm_to_use = LogNorm() if use_log_scale else None
        label_suffix = '(对数尺度)' if use_log_scale else '(线性尺度)'
        x_flat = np.tile(x_vals, y_traj.shape[0])
        y_flat = y_traj.flatten()
        w_flat = np.repeat(weights, y_traj.shape[1])
        mask = np.isfinite(y_flat)
        x_bins = np.linspace(p['x0'], p['x_end'], p['density_bins_x'])
        y_bins = np.linspace(0, p['y_end'], p['density_bins_y'])
        hist, _, _ = np.histogram2d(x_flat[mask], y_flat[mask], bins=[x_bins, y_bins], weights=w_flat[mask])
        hist_masked = np.ma.masked_where(hist == 0, hist)
        
        density_extent = [x_bins[0], x_bins[-1], y_bins[0], y_bins[-1]]
        im_density = ax_density.imshow(hist_masked.T, norm=norm_to_use, cmap='inferno', interpolation='antialiased',
                                       origin='lower', extent=density_extent, aspect='auto')
        ax_density.set_title('该批次軌跡總密度分布 (加权)'); ax_density.set_xlabel('x (m)'); ax_density.set_ylabel('y (m)')
        ax_density.set_facecolor('black')
        cbar_density = fig.colorbar(im_density, ax=ax_density, orientation='vertical')
        cbar_density.set_label('加权粒子密度 ' + label_suffix)

    def _draw_beam_snapshot_plot(self, fig, snapshot_data):
        p = self.params
        fig.suptitle('連續束流瞬時快照分析', fontsize=28, y=0.98)
        
        use_log_scale = p.get('use_log_scale_plots', True)
        norm_to_use = LogNorm() if use_log_scale else None
        label_suffix = '(对数尺度)' if use_log_scale else '(线性尺度)'
        
        mask = np.isfinite(snapshot_data['x']) & np.isfinite(snapshot_data['y']) & np.isfinite(snapshot_data['p'])
        x, y, p_y, w = (snapshot_data[k][mask] for k in ['x', 'y', 'p', 'w'])

        # Left Panel: Spatial Density
        ax1 = fig.add_subplot(1, 2, 1)
        x_bins = np.linspace(p['x0'], p['x_end'], p['density_bins_x'])
        y_bins = np.linspace(0, p['y_end'], p['density_bins_y'])
        hist, _, _ = np.histogram2d(x, y, bins=[x_bins, y_bins], weights=w)
        hist_masked = np.ma.masked_where(hist == 0, hist)
        
        extent1 = [x_bins[0], x_bins[-1], y_bins[0], y_bins[-1]]
        im1 = ax1.imshow(hist_masked.T, norm=norm_to_use, cmap='inferno', interpolation='antialiased',
                         origin='lower', extent=extent1, aspect='auto')
        
        ax1.set_title('瞬時空間密度分布 ρ(x, y)'); ax1.set_xlabel('x (m)'); ax1.set_ylabel('y (m)')
        ax1.set_aspect('equal', adjustable='box'); ax1.set_facecolor('black')
        cbar1 = fig.colorbar(im1, ax=ax1, orientation='vertical')
        cbar1.set_label('瞬时粒子密度 ' + label_suffix)

        # Right Panel: Phase Space Density
        ax2 = fig.add_subplot(1, 2, 2)
        y_bins_phase = np.linspace(0, p['y_end'], 250)
        p_max = np.percentile(np.abs(p_y), 99.5) * 1.2 if p_y.size > 0 else 1.0
        p_bins_phase = np.linspace(-p_max, p_max, 250)
        hist_p, _, _ = np.histogram2d(y, p_y, bins=[y_bins_phase, p_bins_phase], weights=w)
        hist_p_masked = np.ma.masked_where(hist_p == 0, hist_p)
        
        extent2 = [y_bins_phase[0], y_bins_phase[-1], p_bins_phase[0], p_bins_phase[-1]]
        im2 = ax2.imshow(hist_p_masked.T, norm=norm_to_use, cmap='inferno', interpolation='antialiased',
                         origin='lower', extent=extent2, aspect='auto')
        
        ax2.set_title('瞬時相空間分布 ρ(y, p_y)'); ax2.set_xlabel('y (m)'); ax2.set_ylabel('动量 p_y')
        ax2.set_facecolor('black')
        cbar2 = fig.colorbar(im2, ax=ax2, orientation='vertical')
        cbar2.set_label('瞬时相空间密度 ' + label_suffix)
        
        fig.tight_layout(rect=[0, 0, 1, 0.95])