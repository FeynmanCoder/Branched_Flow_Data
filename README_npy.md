1. ### 如何使用新的資料夾選擇和 NPZ 檔案功能

   您的操作變得更加靈活方便。

   1. 打開終端機，確保您在專案根目錄下。

   2. **瀏覽指定的資料夾**：使用 `-f` 或 `--folder` 參數。

      - **只想看 `trainA` (力場檔案)**：

        ```bash
        python visualize_npy.py -f ai_training_data/trainA
        ```

        腳本將只會列出 `trainA` 資料夾內的 `.npy` 和 `.npz` 檔案供您選擇。

      - **只想看 `trainB` (粒子軌跡檔案)**：

        ```bash
        python visualize_npy.py -f ai_training_data/trainB
        ```

   3. **瀏覽所有資料夾 (預設行為)**： 如果您不提供 `-f` 參數，它會掃描整個 `ai_training_data` 目錄。

      ```bash
      python visualize_npy.py
      ```

   4. **直接視覺化單一檔案 (原有功能)**： 這個功能依然保留，且現在同時支援 `.npy` 和 `.npz` 檔案。

      ```bash
      python visualize_npy.py ai_training_data/trainA/group_0000_batch_0001.npy
      ```

   5. **視覺化 NPZ 檔案**：
      - 如果您選擇一個 `.npz` 檔案，腳本會自動檢查其中的內容。
      - **如果檔案內有多個陣列**，它會顯示一個選單，讓您選擇要視覺化哪一個。
        ```
        在 'your_file.npz' 中找到多個陣列:
          [1] potential
          [2] density
        請選擇要視覺化的陣列編號 (1-2):
        ```
      - **如果只有一個陣列**，它會直接顯示該陣列。

   這個修改讓您可以根據需要，快速篩選和定位到您感興趣的數據類型，並能方便地查看 `.npz` 存檔中的具體內容，大大提升了檢查數據的效率。