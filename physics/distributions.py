# branched_flow/physics/distributions.py

import numpy as np

# ==============================================================================
# 1. 定义分布模型
# ==============================================================================

def _generate_momentum(p_y_config: dict, num_particles: int, rng: np.random.Generator) -> np.ndarray:
    """根据配置生成初始动量分布。"""
    config_type = p_y_config.get('type', 'constant')
    params = p_y_config.get('params', {})
    
    if config_type == 'constant':
        return np.full(num_particles, params.get('value', 0.0))
    elif config_type == 'gaussian':
        return rng.normal(loc=params.get('mean', 0.0), scale=params.get('std_dev', 1.0), size=num_particles)
    elif config_type == 'two_way':
        value = params.get('value', 1.0)
        p0 = np.full(num_particles, value)
        indices = rng.choice(num_particles, num_particles // 2, replace=False)
        p0[indices] *= -1
        return p0
    
    print(f"警告：未知的动量分布类型 '{config_type}'，将使用 0.0。")
    return np.zeros(num_particles)

def uniform_distribution(y_end: float, num_particles: int, particle_weight: float, p_y_config: dict, seed: int = None, **kwargs) -> tuple:
    """生成均匀分布的粒子。"""
    y0 = np.linspace(0, y_end, num_particles, endpoint=False)
    rng = np.random.default_rng(seed)
    p0 = _generate_momentum(p_y_config, num_particles, rng)
    w0 = np.full(num_particles, particle_weight)
    return y0, p0, w0

def gaussian_distribution(y_end: float, num_particles: int, particle_weight: float, mean: float, std_dev: float, p_y_config: dict, seed: int = None, **kwargs) -> tuple:
    """生成高斯分布的粒子。"""
    rng = np.random.default_rng(seed)
    y0 = rng.normal(loc=mean, scale=std_dev, size=num_particles) % y_end
    p0 = _generate_momentum(p_y_config, num_particles, rng)
    w0 = np.full(num_particles, particle_weight)
    return y0, p0, w0

# ==============================================================================
# 2. 注册模型
# ==============================================================================

distribution_model_registry = {
    'uniform': uniform_distribution,
    'gaussian': gaussian_distribution,
}

# ==============================================================================
# 3. 定义配置
# ==============================================================================

distribution_configs = {
    'gaussian_thermal_source': {
        'provider': 'gaussian',
        'args': {
            'mean': lambda p: p['y_end'] / 2.0, 
            'std_dev': 2.0,
            'seed': 123,
            'p_y_config': {'type': 'gaussian', 'params': {'mean': 0.0, 'std_dev': 1e-6}}
        },
        'description': "粒子從中心高斯釋放，且初始動量也服從高斯分佈（熱源）。"
    },
    'uniform_default': {
        'provider': 'uniform',
        'args': {
            'p_y_config': {'type': 'constant', 'params': {'value': 0.0}}
        },
        'description': "標準的均勻分佈，粒子在y方向等間隔釋放，初始動量為零。"
    }
}
