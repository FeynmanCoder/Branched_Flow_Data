# branched_flow/analysis/io.py

import os
import numpy as np
from typing import Dict, Any

def save_data_pair(params: Dict[str, Any], batch_num: int, potential_field, y_trajectory, weights, xp):
    """
    高效儲存 AI 訓練所需的一對資料 (勢能場 -> 粒子密度)。
    """
    # --- 1. 計算粒子密度 ---
    bins_y = params['density_bins_y']
    bins_x = params['density_bins_x']
    y_end = params['y_end']
    x_end = params['x_end']
    
    # 使用 xp.histogram2d 進行高效的二維直方圖計算
    density, _, _ = xp.histogram2d(
        y_trajectory.ravel(),
        xp.indices(y_trajectory.shape)[1].ravel() * params['dx'],
        bins=[bins_y, bins_x],
        range=[[0, y_end], [0, x_end]],
        weights=xp.repeat(weights, y_trajectory.shape[1])
    )
    
    # --- 2. 準備儲存路徑 ---
    base_path = params['ai_data_output_path']
    split = params.get('dataset_split', 'train') # 'train', 'validation', or 'test'
    sequence_id = params.get('sequence_id', 0)
    
    # 檔名格式: sequence_XXXX_batch_YYYY.npy
    filename = f"sequence_{sequence_id:04d}_batch_{batch_num:03d}.npy"
    
    path_A = os.path.join(base_path, f"{split}A", filename) # 勢能場
    path_B = os.path.join(base_path, f"{split}B", filename) # 粒子密度
    
    # --- 3. 轉換為 float32 並儲存 ---
    # 將 GPU 陣列轉換為 CPU numpy 陣列以便儲存
    potential_cpu = xp.asnumpy(potential_field).astype(np.float32)
    density_cpu = xp.asnumpy(density).astype(np.float32)
    
    try:
        np.save(path_A, potential_cpu)
        np.save(path_B, density_cpu)
    except IOError as e:
        print(f"警告：寫入 AI 資料失敗 (ID: {sequence_id}, Batch: {batch_num}): {e}")
