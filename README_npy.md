1. ### 如何使用新的資料夾選擇功能

   您的操作變得更加靈活方便。

   1. 打開終端機，確保您在專案根目錄 `train_data` 下。

   2. **瀏覽指定的資料夾**：使用 `-f` 或 `--folder` 參數。

      - **只想看 `trainA` (力場檔案)**：

        Bash

        ```
        python visualize_npy.py -f ai_training_data/trainA
        ```

        腳本將只會列出 `trainA` 資料夾內的 `.npy` 檔案供您選擇。

      - **只想看 `trainB` (粒子軌跡檔案)**：

        Bash

        ```
        python visualize_npy.py -f ai_training_data/trainB
        ```

        腳本將只會列出 `trainB` 資料夾內的 `.npy` 檔案。

   3. **瀏覽所有資料夾 (預設行為)**： 如果您不提供 `-f` 參數，它的行為和以前一樣，會掃描整個 `ai_training_data` 目錄。

      Bash

      ```
      python visualize_npy.py
      ```

   4. **直接視覺化單一檔案 (原有功能)**： 這個功能依然保留。如果您提供了檔案路徑，它會直接顯示該檔案，跳過互動模式。

      Bash

      ```
      python visualize_npy.py ai_training_data/trainA/group_0000_batch_0001.npy
      ```

   這個修改讓您可以根據需要，快速篩選和定位到您感興趣的數據類型，大大提升了檢查數據的效率