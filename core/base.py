# branched_flow/core/base.py

from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple
import numpy as np

class Backend(ABC):
    """
    后端抽象基类。
    它定义了所有具体后端（CPU, GPU）必须实现的通用接口，
    但自身不包含任何具体实现。
    """
    def __init__(self, params: Dict[str, Any], field_data_obj):
        self.params = params
        self.field_data_obj = field_data_obj
        self.xp = None      # 将由子类设置为 numpy 或 cupy
        self.dtype = None   # 将由子类设置为 float32 或 float64
        self._setup_backend_specifics()

    @abstractmethod
    def _setup_backend_specifics(self):
        """设置后端特定的属性，如 self.xp 和 self.dtype。"""
        pass

    @abstractmethod
    def setup_computation(self, num_particles: int):
        """为模拟分配内存并编译任何必要的计算核心。"""
        pass

    @abstractmethod
    def run_single_batch(self, force_field, y0, p0, w0) -> Tuple[Any, Any, Any]:
        """运行一个批次的模拟，并返回力场修正和最终轨迹。"""
        pass

    @abstractmethod
    def get_snapshot_slice(self, y_traj, p_traj, x_target, x_width) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """从轨迹数据中提取一个带有宽度的状态切片。"""
        pass
