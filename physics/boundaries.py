# branched_flow/physics/boundaries.py

import numpy as np
try:
    from numba import cuda
except ImportError:
    cuda = None

# ==============================================================================
# 1. 定义核心逻辑
# ==============================================================================

def _periodic_logic(y, p_y, y_min, y_end):
    if y >= y_end or y < y_min:
        y = y_min + (y - y_min) % (y_end - y_min)
    return y, p_y

def _reflecting_logic(y, p_y, y_min, y_end):
    if y > y_end:
        y, p_y = y_end - (y - y_end), -p_y
    elif y < y_min:
        y, p_y = y_min + (y_min - y), -p_y
    return y, p_y

def _kill_logic(y, p_y, y_min, y_end):
    if y >= y_end or y < y_min:
        y, p_y = np.nan, np.nan
    return y, p_y

# ==============================================================================
# 2. 注册与适配
# ==============================================================================

boundary_condition_registry_cpu = {
    'periodic': _periodic_logic,
    'reflecting': _reflecting_logic,
    'kill': _kill_logic,
}

# Numba JIT compile the same logic for GPU
if cuda:
    boundary_condition_registry_gpu = {
        'periodic': cuda.jit(_periodic_logic, device=True),
        'reflecting': cuda.jit(_reflecting_logic, device=True),
        'kill': cuda.jit(_kill_logic, device=True),
    }
else:
    boundary_condition_registry_gpu = {}
