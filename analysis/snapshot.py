# branched_flow/analysis/snapshot.py

import numpy as np
from typing import List, Dict

def assemble_snapshot_from_slices(snapshot_slices: List[Dict[str, np.ndarray]]) -> Dict[str, np.ndarray]:
    """
    从内存中的状态切片列表中，组装出最终的束流快照。
    """
    print("  [分析] 正在從內存中的狀態切片組裝瞬時束流快照...")
    
    # 检查是否有任何切片可供处理
    if not snapshot_slices:
        print("    - 警告：沒有可用的狀態切片，無法組裝束流圖。")
        return {"x": np.array([]), "y": np.array([]), "p": np.array([]), "w": np.array([])}

    # 使用字典推导式和 np.concatenate 高效拼接所有数据
    # 这会遍历第一个切片的所有键 ('x', 'y', 'p', 'w')
    # 然后为每个键，将所有切片中对应的数组拼接起来
    all_data = {key: np.concatenate([s[key] for s in snapshot_slices]) for key in snapshot_slices[0]}
        
    print(f"  [組裝] 成功從 {len(snapshot_slices)} 個切片中組裝了 {len(all_data['x'])} 個粒子。")
    return all_data
