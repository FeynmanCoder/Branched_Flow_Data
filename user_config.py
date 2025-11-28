# branched_flow/user_config.py

import numpy as np

# ==============================================================================
# 用戶常用配置 (User Configuration)
# 您可以在此處快速調整模擬和生成的關鍵參數
# ==============================================================================

# --- 環境設定 ---
# 設為 True，將使用超算平台的路徑；設為 False，將使用本地路徑。
is_hpc_environment = False

# --- 物理空間尺寸 ---
x_end = 20.0
y_end = 20.0

# --- 勢能場網格步長 (決定 AI 圖片的基礎解析度) ---
dx_potential = 0.025
dy_potential = 0.025

# --- AI 圖片尺寸自動同步開關 ---
# 設為 True: 將強制 density_bins_x/y 與物理網格尺寸一致，確保輸入和輸出的圖片大小相同。
# 設為 False: 您可以手動設定下面的 density_bins_x/y，用於其他非 AI 生成的分析任務。
auto_sync_image_size = True

# --- 隨機勢能場參數設定 ---
# 設為 True: 系統將在下面的 [min, max] 範圍內隨機選擇參數
# 設為 False: 系統將使用下面的 DEFAULT 固定值
enable_random_potential_params = True

# 隨機範圍 (當 enable_random_potential_params = True 時生效)
potential_alpha_min = 3.0
potential_alpha_max = 6.0

potential_amplitude_min = 0.05
potential_amplitude_max = 0.3

# 固定預設值 (當 enable_random_potential_params = False 時生效)
potential_alpha_default = 4.0
potential_amplitude_default = 0.1

# --- 資料集生成設定 ---
num_batches_per_sequence = 2
num_train_sequences = 1
num_validation_sequences = 1
num_test_sequences = 1
