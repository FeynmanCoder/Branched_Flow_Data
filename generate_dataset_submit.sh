#!/bin/bash

# ===================================================================
#                      SBATCH 參數設定
# ===================================================================
#SBATCH --job-name=AIDataGen      # 作業名稱
#SBATCH --output=gen_dataset.%j.out # 標準輸出將寫入此檔案 (%j 會被替換為作業 ID)
#SBATCH --partition=GPU40G        # 指定要使用的 GPU 分區
#SBATCH --qos=low                 # 服務品質
#SBATCH --nodes=1                 # 需要 1 個計算節點
#SBATCH --ntasks-per-node=1       # 每個節點上運行 1 個任務
#SBATCH --gres=gpu:1              # 每個節點需要 1 個 GPU
#SBATCH --cpus-per-task=16        # 每個任務需要 16 個 CPU 核心

# ===================================================================
#                      執行環境與命令
# ===================================================================

# --- 1. 初始化計算節點的 Shell 環境 ---
echo "步驟 1: 初始化計算節點 Shell 環境..."
source /etc/profile
echo "------------------------------------------------------"

# --- 2. 打印作業除錯資訊 ---
echo "作業 ID (SLURM_JOB_ID): $SLURM_JOB_ID"
echo "運行節點 (SLURM_NODELIST): $SLURM_NODELIST"
echo "開始時間: $(date)"
echo "------------------------------------------------------"
nvidia-smi
echo "------------------------------------------------------"

# --- 3. 定義並設定工作目錄 ---
# 【已更新】使用您提供的路徑
PROJECT_ROOT="/lustre/home/2400011491/work/ai_branched_flow/Branched_Flow_Data"
echo "步驟 2: 設定工作目錄..."
cd $PROJECT_ROOT
echo "目前工作目錄: $(pwd)"
echo "------------------------------------------------------"

# --- 4. 載入必要的環境模組 ---
echo "步驟 3: 載入 CUDA 模組..."
# 請根據您叢集的實際情況，確認 CUDA 版本是否需要修改
module load cuda/12.6.0
module list
echo "------------------------------------------------------"

# --- 5. 初始化並啟用您個人的 Conda 環境 ---
echo "步驟 4: 啟用 Conda 環境 'ai_bf'..."
# 這一行會初始化您自己安裝的 Conda
source /lustre/home/2400011491/software/miniconda3/etc/profile.d/conda.sh
# 【已更新】啟用您指定的 ai_bf 環境
conda activate ai_bf
echo "Conda 環境 'ai_bf' 已啟用。"
echo "------------------------------------------------------"

# --- 6. 驗證 Python 環境 ---
echo "步驟 5: 驗證當前 Python 環境..."
echo "==> 當前使用的 Python 直譯器是: $(which python)"
echo "------------------------------------------------------"

# --- 7. 執行 AI 資料集生成腳本 ---
echo "步驟 6: 開始運行 AI 資料集生成..."
# 【已更新】直接執行 main.py 並指定 generate 模式和輸出路徑
python main.py --mode generate --ai-data-output-path /lustre/home/2400011491/data/ai_train_data/new_data
echo "------------------------------------------------------"

echo "資料集生成完成。"
echo "結束時間: $(date)"
echo "作業執行完畢。"

# --- 8. 退出 Conda 環境 ---
conda deactivate