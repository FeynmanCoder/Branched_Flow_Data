# branched_flow/analysis/io.py

import os
import numpy as np
from typing import Dict, Any, List

class HistoryManager:
    """管理模拟历史快照的记录和检索。"""
    def __init__(self, params: Dict[str, Any]):
        self.plot_interval = params.get('batch_plot_interval', 1)
        self.simulation_history: List[Dict[str, Any]] = []

    def should_snapshot(self, batch_num: int, total_batches: int) -> bool:
        """判断当前批次是否需要记录详细快照。"""
        is_first = batch_num == 1
        is_last = batch_num == total_batches
        is_interval = (batch_num > 1) and ((batch_num - 1) % self.plot_interval == 0)
        return is_first or is_last or is_interval

    def record_snapshot(self, snapshot_data: Dict[str, Any]):
        """将一个批次的快照数据添加到历史记录中。"""
        self.simulation_history.append(snapshot_data)

    def get_history(self) -> List[Dict[str, Any]]:
        """返回完整的模拟历史记录。"""
        return self.simulation_history

class ExportManager:
    """负责将模拟的原始数据导出到磁盘。"""
    def __init__(self, params: Dict[str, Any], xp):
        self.is_enabled = params.get('enable_export', False)
        self.export_path = params.get('default_export_path')
        self.xp = xp # cupy or numpy module
        if self.is_enabled:
            if not os.path.isabs(self.export_path):
                self.export_path = os.path.join(os.getcwd(), self.export_path)
            os.makedirs(self.export_path, exist_ok=True)
            print(f"  [數據導出] 功能已啟用。數據將被保存到: '{self.export_path}'")

    def export_batch_data(self, snapshot_data: Dict[str, Any]):
        """将单个批次的快照数据导出为 .npz 文件。"""
        if not self.is_enabled:
            return
        
        batch_num = snapshot_data['batch_num']
        file_path = os.path.join(self.export_path, f"report_snapshot_batch_{batch_num:05d}.npz")
        
        # 准备数据以便保存 (将GPU数组转换为CPU numpy数组)
        data_to_save = {
            k: v for k, v in snapshot_data.items() if not isinstance(v, self.xp.ndarray)
        }
        data_to_save['field_before_batch'] = self.xp.asnumpy(snapshot_data['field_before_batch'])
        data_to_save['y_reshaped'] = self.xp.asnumpy(snapshot_data['y_reshaped'])
        data_to_save['p_reshaped'] = self.xp.asnumpy(snapshot_data['p_reshaped'])
        
        try:
            np.savez_compressed(file_path, **data_to_save)
        except IOError as e:
            print(f"警告：寫入文件 {file_path} 失敗: {e}")
