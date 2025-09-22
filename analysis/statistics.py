# branched_flow/analysis/statistics.py

import numpy as np
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
from typing import Dict, Any

class StatisticsManager:
    """负责计算模拟过程中的所有统计数据。"""
    def __init__(self, params: Dict[str, Any], xp):
        self.params = params
        self.xp = xp
        self.dx_potential = params['dx_potential']
        self.dy_potential = params['dy_potential']

    def calculate_batch_stats(self, force_field, y_traj, p_traj, perform_full_analysis: bool) -> Dict[str, float]:
        p = self.params
        final_p_squared = self.xp.asnumpy(p_traj[:, -1])**2
        
        stats = {
            "total_kinetic_energy": np.sum(final_p_squared[np.isfinite(final_p_squared)]) / (2 * p['m']),
            "mean_field_val": self.xp.mean(force_field).item(),
            "std_field_val": self.xp.std(force_field).item(),
            "correlation_length": np.nan,
            "focusing_length": np.nan,
        }

        if perform_full_analysis:
            field_cpu = self.xp.asnumpy(force_field)
            stats["correlation_length"] = self._calculate_correlation_length(field_cpu)
            stats["focusing_length"] = self._calculate_focus_by_scintillation(y_traj)
        
        return stats

    def _calculate_correlation_length(self, field_matrix: np.ndarray) -> float:
        ny, nx = field_matrix.shape
        all_acfs = []
        for i in range(nx):
            col = field_matrix[:, i].astype(np.float64)
            col -= np.mean(col)
            if np.var(col) < 1e-9: continue
            
            fourier = np.fft.fft(col)
            acf = np.fft.ifft(np.abs(fourier)**2).real
            acf = np.fft.fftshift(acf)
            all_acfs.append(acf / acf[ny // 2])

        if not all_acfs: return np.nan
        
        avg_acf = np.mean(np.array(all_acfs), axis=0)
        lags_y = (np.arange(ny) - ny // 2) * self.dy_potential
        
        center_idx = ny // 2
        lags_fit = lags_y[center_idx:]
        acf_fit = avg_acf[center_idx:]

        def gaussian_fit(y, Lc):
            return np.exp(-(y**2) / (max(abs(Lc), 1e-9)**2))

        try:
            popt, _ = curve_fit(gaussian_fit, lags_fit, acf_fit, p0=[1.0])
            return abs(popt[0])
        except Exception:
            return np.nan

    def _calculate_focus_by_scintillation(self, y_trajectories) -> float:
        p = self.params
        y_cpu = self.xp.asnumpy(y_trajectories)
        x_vals = np.arange(p['x0'], p['x_end'], p['dx'])
        
        x_flat = np.tile(x_vals, y_cpu.shape[0])
        y_flat = y_cpu.flatten()
        
        valid_mask = np.isfinite(y_flat)
        x_bins = np.linspace(p['x0'], p['x_end'], p['density_bins_x'])
        y_bins = np.linspace(0, p['y_end'], p['density_bins_y'])
        
        rho, _, _ = np.histogram2d(x_flat[valid_mask], y_flat[valid_mask], bins=[x_bins, y_bins])
        rho = rho.T
        
        mean_I = np.mean(rho, axis=0)
        mean_I2 = np.mean(rho**2, axis=0)
        
        valid = mean_I > 1e-9
        scint_factor = np.zeros_like(mean_I)
        scint_factor[valid] = mean_I2[valid] / (mean_I[valid]**2) - 1
        
        x_centers = (x_bins[:-1] + x_bins[1:]) / 2
        start_idx = int(0.1 * len(x_centers))
        
        peaks, props = find_peaks(
            scint_factor[start_idx:],
            height=p['peak_finding_height_ratio'] * np.max(scint_factor),
            prominence=p['peak_finding_prominence_ratio'] * np.max(scint_factor)
        )
        
        if len(peaks) > 0:
            return x_centers[start_idx + peaks[np.argmax(props['prominences'])]]
        return np.nan