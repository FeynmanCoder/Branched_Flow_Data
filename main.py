# branched_flow/main.py

import sys
import os
import argparse
import shutil
from timer import SimpleTimer

# --- 專案路徑設定 ---
_current_file_dir = os.path.dirname(os.path.abspath(__file__))
if _current_file_dir not in sys.path:
    sys.path.insert(0, _current_file_dir)

# --- 匯入模組 ---
from config import get_config
from simulation import Simulation
from simulation_setup import setup_simulation_environment

def _prepare_output_directory(path: str, clean: bool = True):
    """準備輸出目錄：如果 clean=True 則清空，然後確保它存在。"""
    if clean and os.path.exists(path):
        print(f"清空已存在的輸出目錄 '{path}'...")
        shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)
    print(f"輸出目錄 '{path}' 已準備就緒。")

def generate_dataset(params: dict):
    """高效生成包含演化序列的 AI 訓練、驗證和測試資料集。"""
    num_train = params['num_train_sequences']
    num_val = params['num_validation_sequences']
    num_test = params['num_test_sequences']
    num_total_sequences = num_train + num_val + num_test
    num_batches_per_sequence = params['num_batches_per_sequence']
    
    print(f"\n--- AI 資料集生成任務 ---")
    print(f"任務配置: {num_train} 訓練序列, {num_val} 驗證序列, {num_test} 測試序列 (共 {num_total_sequences} 個)。")
    print(f"每個序列包含 {num_batches_per_sequence} 個批次。")
    
    output_path = params['ai_data_output_path']
    _prepare_output_directory(output_path)
    
    # 為不同資料集建立子目錄
    for split in ['train', 'validation', 'test']:
        for folder in [f'{split}A', f'{split}B']:
            os.makedirs(os.path.join(output_path, folder), exist_ok=True)
    
    print(f"所有資料將儲存於 '{output_path}' 的對應子目錄中。")
    
    timer = SimpleTimer()
    timer.start("總任務")
    
    with timer.record("初始化與編譯"):
        print("\n正在進行一次性初始化並編譯 GPU 核心，請稍候...")
        # 使用一個固定的、簡單的勢場來觸發編譯
        init_params = params.copy()
        init_params['potential_config_name'] = 'sinusoidal_default'
        initial_field, initial_particles, updated_params = setup_simulation_environment(init_params)
        sim = Simulation(updated_params, initial_field, initial_particles)
        sim.set_timer(timer)
        print("初始化與編譯完成！\n")
    
    sequence_counter = 0
    for split_name, num_sequences_in_split in [('train', num_train), ('validation', num_val), ('test', num_test)]:
        if num_sequences_in_split == 0:
            continue
        
        print(f"\n--- 開始生成 {split_name} 資料集 ({num_sequences_in_split} 個序列) ---")
        for i in range(num_sequences_in_split):
            sequence_counter += 1
            print(f"\n--- 正在生成第 {sequence_counter}/{num_total_sequences} 個序列 (屬於 {split_name} 集) ---")
            
            with timer.record(f"第 {sequence_counter} 個序列模擬"):
                sim.reset_for_new_sequence(seed=sequence_counter) 
                sim.params['dataset_split'] = split_name
                sim.params['sequence_id'] = sequence_counter
                # 在此模式下，run 函數只負責計算和儲存 AI 資料
                sim.run(num_batches=num_batches_per_sequence)

    timer.stop("總任務")
    print("\n序列資料集生成完畢！")
    timer.report()

def main():
    """程式主入口，解析命令行參數並啟動資料生成。"""
    parser = argparse.ArgumentParser(
        description="一個用於高效生成分支流 AI 資料集的程式。",
        formatter_class=argparse.RawTextHelpFormatter
    )
    
    base_params = get_config()
    print("提示：已載入 config.py 中的預設設定。")

    # 動態地為所有可配置參數添加命令行接口
    for key, value in base_params.items():
        arg_type = type(value)
        if arg_type is bool:
            # 對於布林值，使用 action='store_true' 或 'store_false'
            # 這裡簡化處理，假設大部分布林開關是為了覆寫為 True
            parser.add_argument(f'--{key.replace("_", "-")}', action=argparse.BooleanOptionalAction, default=None)
        else:
            parser.add_argument(f'--{key.replace("_", "-")}', type=arg_type, default=None, help=f'覆寫 {key} 參數 (預設: {value})')
    
    args = parser.parse_args()
    
    # 將命令行參數覆寫到基礎設定上
    final_params = base_params.copy()
    for key, value in vars(args).items():
        if value is not None:
            final_params[key] = value

    # 直接執行資料生成任務
    generate_dataset(final_params)

if __name__ == "__main__":
    main()
