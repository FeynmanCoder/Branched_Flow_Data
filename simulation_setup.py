# branched_flow/simulation_setup.py

import numpy as np
from typing import Dict, Any, Tuple

from core.field import FieldData
from physics import potentials, interactions, distributions, updates, boundaries

def _validate_configs(params: Dict[str, Any]):
    is_quiet = params.get('quiet_mode', False)
    if not is_quiet: print("--- 驗證配置信息 ---")
    
    config_map = {
        'potential_config_name': (potentials.potential_configs, "勢能"),
        'interaction_config_name': (interactions.interaction_configs, "相互作用"),
        'distribution_config_name': (distributions.distribution_configs, "粒子分布"),
        'potential_update_rule': (updates.update_rule_registry, "力场更新规则"),
        'boundary_condition': (boundaries.boundary_condition_registry_cpu.keys(), "边界条件")
    }
    for name, (registry, desc) in config_map.items():
        if params.get(name) not in registry:
            raise ValueError(f"错误: {desc}配置 '{params.get(name)}' 不存在。可用: {list(registry)}")
    
    if not is_quiet: print("配置验证通过。")

def _load_interaction_config(params: Dict[str, Any]) -> Dict[str, Any]:
    config = interactions.interaction_configs[params['interaction_config_name']]
    params.update(config)
    return params

def _setup_potential(params: Dict[str, Any]) -> FieldData:
    is_quiet = params.get('quiet_mode', False)
    if not is_quiet: print("--- 初始化勢能場 ---")
    
    p_conf = potentials.potential_configs[params['potential_config_name']]
    provider = potentials.potential_source_registry[p_conf['provider']]
    
    provider_args = p_conf.get('args', {}).copy()
    if 'potential_seed_override' in params:
        new_seed = params['potential_seed_override']
        provider_args['seed'] = new_seed
        if not is_quiet: print(f"  [势场生成] 關鍵修正：已使用來自控制腳本的唯一種子: {new_seed}")

    args = {
        'x_start': params['x0'], 'x_end_grid': params['x_end'], 
        'y_start': 0.0, 'y_end_grid': params['y_end'], 
        'dx': params['dx_potential'], 'dy': params['dy_potential'], 
        **provider_args
    }
    matrix = provider(**args)
    return FieldData(matrix, params['x0'], 0.0, params['dx_potential'], params['dy_potential'])

def _setup_particle_distribution(params: Dict[str, Any]) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
    if not params.get('quiet_mode', False): print("--- 初始化粒子分佈 ---")
    p = params
    d_conf = distributions.distribution_configs[p['distribution_config_name']]
    provider = distributions.distribution_model_registry[d_conf['provider']]
    
    args = {k: v(p) if callable(v) else v for k, v in d_conf.get('args', {}).items()}
    
    all_args = {
        'y_end': p['y_end'], 
        'num_particles': p['num_particles'],
        'particle_weight': p['particle_weight'],
        **args
    }
    y0, p0, w0 = provider(**all_args)
    
    if len(y0) != p['num_particles']:
        p['num_particles'] = len(y0)
        if not params.get('quiet_mode', False): print(f"    - 粒子数已从 {p['num_particles']} 更新为 {len(y0)}")

    return {'y0': y0, 'p0': p0, 'w0': w0}, p

def convert_potential_to_force_field(matrix: np.ndarray, dy: float, is_quiet: bool = False) -> np.ndarray:
    if not is_quiet: print("  [Field Conversion] Converting Potential(V) to Force Field(Fy)...")
    fy = np.zeros_like(matrix)
    fy[1:-1, :] = - (matrix[2:, :] - matrix[:-2, :]) / (2 * dy)
    fy[0, :] = - (matrix[1, :] - matrix[0, :]) / dy
    fy[-1, :] = - (matrix[-1, :] - matrix[-2, :]) / dy
    return fy

def setup_simulation_environment(params: Dict[str, Any]) -> Tuple[FieldData, Dict[str, np.ndarray], Dict[str, Any]]:
    is_quiet = params.get('quiet_mode', False)
    _validate_configs(params)
    params = _load_interaction_config(params)
    
    potential_obj = _setup_potential(params)
    force_matrix = convert_potential_to_force_field(potential_obj.get_field_matrix(), params['dy_potential'], is_quiet)
    initial_field = FieldData(force_matrix, params['x0'], 0.0, params['dx_potential'], params['dy_potential'])
    
    initial_particles, updated_params = _setup_particle_distribution(params)
    
    return initial_field, initial_particles, updated_params