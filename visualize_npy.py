import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import sys

def visualize_file_data(filepath: str, title: str = None, cmap: str = 'viridis', use_log_scale: bool = False, save_path: str = None):
    """
    載入一個 .npy 或 .npz 檔案並將其視覺化為熱圖。
    如果檔案是 .npz 且包含多個陣列，會提示使用者選擇要視覺化的陣列。
    """
    # --- 1. 檢查檔案是否存在 ---
    if not os.path.exists(filepath):
        print(f"錯誤：找不到檔案 '{filepath}'")
        return

    data = None
    selected_array_name = None

    # --- 2. 根據檔案類型載入資料 ---
    try:
        file_ext = os.path.splitext(filepath)[1]
        
        if file_ext == '.npy':
            data = np.load(filepath)
            selected_array_name = "N/A (npy file)"
        elif file_ext == '.npz':
            npz_file = np.load(filepath)
            arrays = npz_file.files
            
            if not arrays:
                print(f"錯誤：.npz 檔案 '{filepath}' 中不包含任何陣列。")
                return
            
            if len(arrays) == 1:
                selected_array_name = arrays[0]
                data = npz_file[selected_array_name]
            else:
                # 如果有多個陣列，提示使用者選擇
                print(f"\n在 '{os.path.basename(filepath)}' 中找到多個陣列:")
                for i, name in enumerate(arrays):
                    print(f"  [{i+1}] {name}")
                
                while True:
                    try:
                        choice = input(f"請選擇要視覺化的陣列編號 (1-{len(arrays)}): ")
                        choice_idx = int(choice) - 1
                        if 0 <= choice_idx < len(arrays):
                            selected_array_name = arrays[choice_idx]
                            data = npz_file[selected_array_name]
                            break
                        else:
                            print("無效的選擇，請重新輸入。")
                    except (ValueError, IndexError):
                        print("無效的輸入，請輸入數字。")
                    except (KeyboardInterrupt, EOFError):
                        print("\n操作已取消。")
                        return
        else:
            print(f"錯誤：不支援的檔案類型 '{file_ext}'。僅支援 .npy 和 .npz。")
            return

    except Exception as e:
        print(f"錯誤：載入檔案 '{filepath}' 失敗: {e}")
        return

    # --- 3. 打印數據資訊 (非常有用於除錯) ---
    print(f"\n--- 檔案資訊: {os.path.basename(filepath)} ---")
    if selected_array_name:
        print(f"  - 正在顯示陣列: '{selected_array_name}'")
    print(f"  - 陣列維度 (Shape): {data.shape}")
    print(f"  - 資料類型 (dtype): {data.dtype}")
    print(f"  - 最大值 (Max): {np.max(data):.4f}")
    print(f"  - 最小值 (Min): {np.min(data):.4f}")
    print(f"  - 平均值 (Mean): {np.mean(data):.4f}")
    
    display_data = data
    log_info = ""
    if use_log_scale:
        display_data = np.log1p(data)
        log_info = " (對數尺度)"
        print("  - 已啟用對數縮放 (log(1+x)) 進行視覺化")
    print("-" * (len(os.path.basename(filepath)) + 16))

    # --- 5. 使用 Matplotlib 進行繪圖 ---
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(display_data, cmap=cmap, origin='lower', aspect='auto')
    fig.colorbar(im, ax=ax)
    
    plot_title = title if title else os.path.basename(filepath)
    if selected_array_name and 'npy' not in selected_array_name:
        plot_title += f" - [{selected_array_name}]"

    ax.set_title(plot_title + log_info, fontsize=16)
    ax.set_xlabel("X 軸 (像素)")
    ax.set_ylabel("Y 軸 (像素)")
    fig.tight_layout()

    # --- 6. 顯示或儲存圖像 ---
    if save_path:
        output_dir = os.path.dirname(save_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        fig.savefig(save_path, dpi=150)
        print(f"圖像已儲存至: {save_path}")
    else:
        plt.show()

def interactive_mode(search_path='ai_training_data'):
    """
    掃描指定的 search_path 目錄並提供一個互動式選單來選擇要視覺化的檔案。
    """
    if not os.path.isdir(search_path):
        print(f"錯誤：找不到 '{search_path}' 資料夾。請確保您提供了正確的路徑。")
        return

    print(f"正在掃描 '{search_path}' 中的 .npy 和 .npz 檔案...")
    found_files = []
    # 使用 os.walk 來遞迴掃描所有子資料夾
    for root, _, files in os.walk(search_path):
        for file in sorted(files):
            if file.endswith(('.npy', '.npz')):
                found_files.append(os.path.join(root, file))

    if not found_files:
        print(f"在 '{search_path}' 中找不到任何 .npy 或 .npz 檔案。")
        return

    # 對檔案路徑進行排序，確保每次顯示順序一致
    found_files.sort()

    while True:
        print(f"\n--- 請選擇要視覺化的檔案 (正在瀏覽: {search_path}) ---")
        for i, f_path in enumerate(found_files):
            display_path = os.path.relpath(f_path).replace("\\", "/")
            print(f"  [{i+1}] {display_path}")
        print("  [q] 退出")
        
        try:
            choice = input("請輸入檔案編號 (或 'q' 退出): ")
            if choice.lower() == 'q':
                break
            
            choice_idx = int(choice) - 1
            if 0 <= choice_idx < len(found_files):
                selected_file = found_files[choice_idx]
                
                cmap = 'coolwarm' if 'A' in selected_file else 'inferno'
                log_choice = input("是否使用對數尺度進行視覺化? (y/N): ").lower()
                use_log = log_choice == 'y'
                
                visualize_file_data(selected_file, cmap=cmap, use_log_scale=use_log)
            else:
                print(f"無效的選擇。請輸入 1 到 {len(found_files)} 之間的數字。")

        except ValueError:
            print("無效的輸入，請輸入數字。")
        except (KeyboardInterrupt, EOFError):
            print("\n操作已取消。")
            break

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="一個用於視覺化 .npy 或 .npz 陣列檔案的通用工具。",
        formatter_class=argparse.RawTextHelpFormatter
    )
    
    # 將 filepath 設為可選參數
    parser.add_argument("filepath", type=str, nargs='?', default=None, 
                        help="要直接視覺化的 .npy 或 .npz 檔案的路徑。\n如果留空，則進入互動式瀏覽模式。")
    
    # 新增 --folder 參數
    parser.add_argument("--folder", "-f", type=str, default=None,
                        help="在互動模式下，指定要瀏覽的資料夾路徑。")
                        
    parser.add_argument("--title", type=str, default=None, help="圖像的自訂標題。")
    parser.add_argument("--cmap", type=str, default="viridis", help="Matplotlib 的色彩對映。")
    parser.add_argument("--log", action="store_true", help="啟用此旗標以使用對數尺度。")
    parser.add_argument("--save", type=str, default=None, help="提供路徑以儲存圖像。")
    
    args = parser.parse_args()
    
    if args.filepath:
        # 模式一：如果提供了檔案路徑，則直接視覺化該檔案
        visualize_file_data(args.filepath, args.title, args.cmap, args.log, args.save)
    else:
        # 模式二：如果未提供檔案路徑，則進入互動模式
        # 如果使用者指定了 --folder，則瀏覽該資料夾，否則使用預設路徑
        folder_to_browse = args.folder if args.folder else 'ai_training_data'
        interactive_mode(search_path=folder_to_browse)