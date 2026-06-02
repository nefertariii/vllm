# 檔案：vllm/v1/kv_cache_interface.py（新增）

@dataclass
class EvictionPolicyConfig:
    """Configuration for block eviction policy."""
    
    policy_type: str = "cost_aware"  # "lru", "lfu", "cost_aware", "hybrid"
    recency_weight: float = 0.3
    frequency_weight: float = 0.4
    cost_weight: float = 0.3
    cost_sensitivity: float = 1.0
    min_idle_time: float = 0.1
    enable_dual_pool: bool = True
    max_common_pool_size: int = 1000
    common_pool_protection_factor: float = 2.0
