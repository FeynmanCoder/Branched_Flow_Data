# branched_flow/physics/updates.py

def linear_accumulation(force_field, modification, params):
    """规则一：最简单的线性累加。"""
    force_field += modification

def relaxation_update(force_field, modification, params):
    """规则二：带松弛因子的更新。"""
    alpha = params.get('relaxation_alpha', 0.1)
    force_field *= (1 - alpha)
    force_field += alpha * modification

update_rule_registry = {
    'accumulate': linear_accumulation,
    'relax': relaxation_update,
}
