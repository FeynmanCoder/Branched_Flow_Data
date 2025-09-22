# branched_flow/main.py

import sys
import os
import argparse
import traceback
import shutil
from tqdm import tqdm
from timer import SimpleTimer

# --- 專案路徑設定 (保持不變) ---
_current_file_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(os.path.dirname(_current_file_dir))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

# --- 匯入模組 ---
from config import get_config_for_mode, ACTIVE_MODE
from simulation import Simulation
from simulation_setup import setup_simulation_environment

def _prepare_output_directory(path: str, clean: bool = True):
    """準備輸出目錄：如果 clean=True 則清空，然後確保它存在。"""
    if clean and os.path.exists(path):
        print(f"清空已存在的輸出目錄 '{path}'...")
        shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)
    print(f"輸出目錄 '{path}' 已準備就緒。")

# ==============================================================================
# 模式一：執行標準的單次模擬
# ==============================================================================
def run_simulation_mode(params: dict):
    """執行詳細的單次物理模擬。"""
    print("\n--- 模式: 標準模擬 (simulate) ---")
    _prepare_output_directory(params['plot_output_path'])
    
    try:
        initial_field, initial_particles, updated_params = setup_simulation_environment(params)
        sim = Simulation(updated_params, initial_field, initial_particles)
        sim.run()
        sim.analyze_and_visualize()
        print("\n標準模擬完成。")
    except Exception as e:
        print(f"\n程式出現未處理的錯誤: {e}")
        traceback.print_exc()

# ==============================================================================
# 模式二：執行高效的 AI 資料集生成
# ==============================================================================
# main.py

def run_generation_mode(params: dict):
    """高效生成包含演化序列的 AI 訓練、驗證和測試資料集。"""
    num_train = params['num_train_groups']
    num_val = params['num_validation_groups']
    num_test = params['num_test_groups']
    num_total_groups = num_train + num_val + num_test
    num_batches_per_group = params['num_batches_per_group']
    
    print(f"\n--- 模式: AI 序列資料集生成 ---")
    print(f"任務配置: {num_train} 組訓練, {num_val} 組驗證, {num_test} 組測試 (共 {num_total_groups} 組)。")
    print(f"每組包含 {num_batches_per_group} 個批次。")
    
    output_path = params['ai_data_output_path']
    _prepare_output_directory(output_path)
    
    # 為不同資料集建立子目錄
    for split in ['train', 'validation', 'test']:
        for folder in [f'{split}A', f'{split}B']:
            os.makedirs(os.path.join(output_path, folder), exist_ok=True)
    
    print(f"所有資料將儲存於 '{output_path}' 的對應子目錄中。")
    
    # --- 建立計時器實例 ---
    timer = SimpleTimer()
    
    timer.start("總任務")
    
    # --- 測量一次性初始化和編譯的時間 ---
    with timer.record("初始化與編譯"):
        print("\n正在進行一次性初始化並編譯 GPU 核心，請稍候...")
        init_params = params.copy()
        init_params['potential_config_name'] = 'sinusoidal_default'
        initial_field, initial_particles, updated_params = setup_simulation_environment(init_params)
        sim = Simulation(updated_params, initial_field, initial_particles)
        sim.set_timer(timer)
        print("初始化與編譯完成！\n")
    
    # --- 外層迴圈：遍歷每一組獨立的模擬 ---
    group_counter = 0
    for split_name, num_groups_in_split in [('train', num_train), ('validation', num_val), ('test', num_test)]:
        if num_groups_in_split == 0:
            continue
        
        print(f"\n--- 開始生成 {split_name} 資料集 ({num_groups_in_split} 組) ---")
        for i in range(num_groups_in_split):
            group_counter += 1
            print(f"\n--- 正在生成第 {group_counter}/{num_total_groups} 組 (屬於 {split_name} 集) ---")
            
            with timer.record(f"第 {group_counter} 組模擬 (總共)"):
                sim.reset_for_new_run(seed=group_counter) 
                sim.params['dataset_split'] = split_name
                sim.params['simulation_id'] = group_counter
                sim.run(num_batches=num_batches_per_group)

    timer.stop("總任務")
    print("\n序列資料集生成完畢！")
    
    timer.report()

# 我們需要在 Simulation 類別中新增一個 reset_for_new_run 方法
# ==============================================================================
# 主入口：解析模式並分派任務
# ==============================================================================
def main():
    """程式主入口，根據模式選擇執行不同任務。"""
    parser = argparse.ArgumentParser(
        description="一個用於分支流物理模擬與 AI 資料集生成的程式。",
        formatter_class=argparse.RawTextHelpFormatter
    )
    
    # 【核心修改】將 config.py 中的 ACTIVE_MODE 作為 argparse 的預設值
    parser.add_argument(
        '--mode',
        type=str,
        default=ACTIVE_MODE,  # <-- 關鍵改動：預設值來自 config.py
        choices=['simulate', 'generate'],
        help=f"選擇程式的運行模式。\n"
             f"如果不在命令列中指定，將使用 config.py 中設定的預設模式: '{ACTIVE_MODE}'"
    )

    temp_args, _ = parser.parse_known_args()
    
    base_params = get_config_for_mode(temp_args.mode)
    print(f"提示：已載入 '{temp_args.mode}' 模式的預設設定。")

    for key, value in base_params.items():
        arg_type = type(value)
        if arg_type is bool:
            parser.add_argument(f'--{key.replace("_", "-")}', action=argparse.BooleanOptionalAction, default=None)
        else:
            parser.add_argument(f'--{key.replace("_", "-")}', type=arg_type, default=None, help=f'覆寫 {key} 參數')
    
    args = parser.parse_args()
    
    final_params = base_params.copy()
    for key, value in vars(args).items():
        if value is not None:
            final_params[key] = value

    if args.mode == 'simulate':
        run_simulation_mode(final_params)
    elif args.mode == 'generate':
        run_generation_mode(final_params)

if __name__ == "__main__":
    main()