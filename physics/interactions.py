# branched_flow/physics/interactions.py

import numpy as np
import math
try:
    from numba import cuda
except ImportError:
    cuda = None

# ==============================================================================
# 1. 定义核心物理逻辑 (Python/NumPy)
# ==============================================================================

def _force_dependent_exp_decay_logic(r, force_at_particle, _, mod_sign, mod_a, mod_b):
    base = mod_a * math.exp(-mod_b * r)
    
    # <--- 核心修正：用 Numba 兼容的 if/else 替换 np.sign ---
    # This simple conditional works for both CPU (NumPy arrays/scalars)
    # and GPU (scalars).
    if force_at_particle >= 0:
        sign = 1.0
    else:
        sign = -1.0
        
    return mod_sign * sign * base

def _gaussian_logic(r, _, __, mod_sign, amplitude, sigma):
    if sigma <= 0: return 0.0
    return mod_sign * amplitude * math.exp(-0.5 * (r / sigma)**2)

# ==============================================================================
# 2. 模型注册与自动适配
# ==============================================================================

interaction_model_registry_cpu = {
    'force_dependent_exp_decay': _force_dependent_exp_decay_logic,
    'gaussian': _gaussian_logic,
}

if cuda:
    interaction_model_registry_gpu = {
        'force_dependent_exp_decay': cuda.jit(_force_dependent_exp_decay_logic, device=True),
        'gaussian': cuda.jit(_gaussian_logic, device=True),
    }
else:
    interaction_model_registry_gpu = {}


# ==============================================================================
# 3. 定义配置
# ==============================================================================

interaction_configs = {
    'stable_feedback': {
        'interaction_model': 'force_dependent_exp_decay',
        'interaction_params': {'mod_sign': -1, 'mod_a': 0.0005, 'mod_b': 5.0},
        'description': "稳定的负反馈模型，粒子尝试抵消其所在位置的力场。"
    },
    'gaussian_force_source': {
        'interaction_model': 'gaussian',
        'interaction_params': {'mod_sign': 1, 'amplitude': 0.1, 'sigma': 0.5},
        'description': "高斯力源模型，每个粒子都会主动产生一个正向的力场。"
    }
}
