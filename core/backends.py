# branched_flow/core/backends.py

from typing import Dict, Any
from .field import FieldData
from .base import Backend

# 导入具体的后端实现以便工厂函数可以使用它们
from .cpu_backend import CPUBackend
from .gpu_backend import GPUBackend

try:
    import cupy as cp
except ImportError:
    cp = None

def get_backend(params: Dict[str, Any], field_data_obj: FieldData) -> Backend:
    """
    后端工厂函数。
    根据配置创建并返回一个具体的后端实例 (CPUBackend 或 GPUBackend)。
    """
    if params['backend'] == 'gpu' and cp is not None:
        return GPUBackend(params, field_data_obj)
    
    if params['backend'] == 'gpu':
        print("警告：请求了GPU后端，但Cupy不可用。将回退到CPU后端。")
        
    return CPUBackend(params, field_data_obj)
