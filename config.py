# branched_flow/config.py

import numpy as np
from user_config import *

# ==============================================================================
# 2. 基礎參數 (所有模式共享)
# ==============================================================================

BASE_PARAMS = {
    # --- 後端與高層配置 ---
    'backend': 'gpu',
    'boundary_condition': 'kill',
    
    # --- 路徑設定 ---
    'default_export_path': 'exported_data',
    'plot_output_path': 'simulation_plots',
    'ai_data_output_path': 'ai_training_data',

    # --- 核心物理與網格參數 (使用上面定義的變數) ---
    'm': 1.0,
    'v_x': 1.0,
    'particle_weight': 1.0,
    'x0': 0.0,
    'x_end': x_end,
    'y_end': y_end,
    'num_particles': 256,
    'dx': 0.01, # 這是積分步長，與圖片尺寸無關
    'dx_potential': dx_potential,
    'dy_potential': dy_potential,
    'interaction_cutoff_radius': 5.0,
    'precision': 'float32',
    
    # --- 密度圖/軌跡圖的畫素桶 (bins) 尺寸 ---
    # 如果上面的開關為 False，程式將會使用這兩個手動設定的值
    'density_bins_y': 500,
    'density_bins_x': 500,
    
    # --- 分析參數 ---
    'peak_finding_height_ratio': 0.1,
    'peak_finding_prominence_ratio': 0.1,
}

# --- 根據開關狀態，有條件地覆寫 density_bins ---
if auto_sync_image_size:
    # 根據物理參數精確計算網格點數 (圖片的像素尺寸)
    num_grid_y = int(round(y_end / dy_potential)) + 1
    num_grid_x = int(round(x_end / dx_potential)) + 1
    
    print(f"--- AI 圖像尺寸自動同步已啟用： {num_grid_y} (高) x {num_grid_x} (寬) pixels ---")
    
    # 將計算出的尺寸應用到基礎參數中
    BASE_PARAMS['density_bins_y'] = num_grid_y
    BASE_PARAMS['density_bins_x'] = num_grid_x


# ==============================================================================
# 模式專用設定 (已簡化為單一模式)
# ==============================================================================
def get_config() -> dict:
    """返回唯一的資料生成設定字典。"""
    
    # --- 根據開關決定路徑 ---
    ai_data_path = '/lustre/home/2400011491/data/ai_train_data' if is_hpc_environment else 'ai_training_data'

    # --- 基礎參數直接整合 ---
    params = {
        # --- 後端與高層配置 ---
        'backend': 'gpu',
        'boundary_condition': 'kill',
        
        # --- 路徑設定 ---
        'default_export_path': 'exported_data', # 雖然不啟用，但保留以防萬一
        'plot_output_path': 'simulation_plots', # 同上
        'ai_data_output_path': ai_data_path,

        # --- 核心物理與網格參數 ---
        'm': 1.0,
        'v_x': 1.0,
        'particle_weight': 1.0,
        'x0': 0.0,
        'x_end': x_end,
        'y_end': y_end,
        'num_particles': 256,
        'dx': 0.01,
        'dx_potential': dx_potential,
        'dy_potential': dy_potential,
        'interaction_cutoff_radius': 5.0,
        'precision': 'float64',
        
        # --- 密度圖/軌跡圖的畫素桶 (bins) 尺寸 ---
        'density_bins_y': 500,
        'density_bins_x': 500,
        
        # --- 分析參數 (保留給 AI 資料生成可能用到的部分) ---
        'peak_finding_height_ratio': 0.1,
        'peak_finding_prominence_ratio': 0.1,

        # --- 物理模型配置 ---
        'potential_config_name': 'smooth_random_gpu',
        'interaction_config_name': 'stable_feedback',
        'distribution_config_name': 'uniform_default',
        'potential_update_rule': 'accumulate',
        
        # --- 關鍵的隨機勢能場參數 ---
        'enable_random_potential_params': enable_random_potential_params,
        
        # 隨機範圍
        'potential_alpha_min': potential_alpha_min,
        'potential_alpha_max': potential_alpha_max,
        'potential_amplitude_min': potential_amplitude_min,
        'potential_amplitude_max': potential_amplitude_max,
        
        # 固定預設值
        'potential_alpha': potential_alpha_default,
        'potential_amplitude': potential_amplitude_default,
        
        # --- 執行效率與資料集設定 ---
        'num_batches_per_sequence': num_batches_per_sequence,
        'use_log_scale_plots': True,
        'quiet_mode': True,

        # --- AI 資料集生成任務的宏觀參數 ---
        'num_train_sequences': num_train_sequences,
        'num_validation_sequences': num_validation_sequences,
        'num_test_sequences': num_test_sequences,
    }

    # --- 根據開關狀態，有條件地覆寫 density_bins ---
    if auto_sync_image_size:
        num_grid_y = int(round(y_end / dy_potential)) + 1
        num_grid_x = int(round(x_end / dx_potential)) + 1
        
        print(f"--- AI 圖像尺寸自動同步已啟用： {num_grid_y} (高) x {num_grid_x} (寬) pixels ---")
        
        params['density_bins_y'] = num_grid_y
        params['density_bins_x'] = num_grid_x
        
    return params