# 檔案：examples/cost_aware_eviction_demo.py

from vllm.v1.core.block_eviction_policy import (
    CostAwareEvictionPolicy,
    HybridCostAwareEvictionPolicy,
)
from vllm.v1.core.kv_cache_manager import KVCacheManager

# 示例 1：基礎 Cost-Aware 策略
eviction_policy = CostAwareEvictionPolicy(
    recency_weight=0.3,    # 30% 權重給最後訪問時間
    access_weight=0.4,     # 40% 權重給訪問頻率
    frequency_weight=0.3,  # 30% 權重給代價（Tokens 數量）
    cost_sensitivity=1.0,  # 代價敏感度
)

# 示例 2：混合 Dual-Pool 策略
eviction_policy = HybridCostAwareEvictionPolicy(
    recency_weight=0.25,
    access_weight=0.35,
    frequency_weight=0.4,
    cost_sensitivity=1.2,
    is_common_pool=False,
    common_pool_protection_factor=2.0,  # Common pool blocks 獲得 2x 保護
)

# 示例 3：初始化 KVCacheManager 使用 cost-aware eviction
kv_cache_manager = KVCacheManager(
    kv_cache_config=config,
    max_model_len=4096,
    hash_block_size=16,
    enable_caching=True,
    enable_dual_pool=True,  # 啟用雙池結構
    enable_cost_aware_eviction=True,  # 啟用代價感知淘汰
    eviction_policy=eviction_policy,
)

# 獲取監控統計
stats = kv_cache_manager.make_eviction_stats()
print(f"Cache Stats: {stats}")
