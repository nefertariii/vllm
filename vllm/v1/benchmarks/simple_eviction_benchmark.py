# 檔案：vllm/v1/benchmarks/simple_eviction_benchmark.py
# SPDX-License-Identifier: Apache-2.0

import time
import json
from dataclasses import dataclass, asdict
from typing import List, Dict
import numpy as np
from datetime import datetime

from vllm.logger import init_logger

logger = init_logger(__name__)


@dataclass
class SimpleMetrics:
    """簡化版的指標。"""
    policy_name: str
    cache_hit_rate: float = 0.0
    cache_miss_rate: float = 0.0
    total_evictions: int = 0
    avg_evicted_tokens: float = 0.0
    total_recompute_tokens: int = 0
    total_time_seconds: float = 0.0


class SimpleEvictionBenchmark:
    """
    簡單的模擬器：
    - 模擬一個簡單的 block cache
    - 測試 LRU vs Cost-Aware eviction
    """
    
    def __init__(self, num_blocks: int = 1000):
        self.num_blocks = num_blocks
    
    def simulate_lru(self, requests: List[Dict]) -> SimpleMetrics:
        """
        模擬原版 LRU eviction。
        """
        metrics = SimpleMetrics(policy_name="LRU (Original)")
        
        cache = {}  # prefix_id -> (block_id, token_count, last_access_time)
        block_id_counter = 0
        cache_hits = 0
        evictions = 0
        evicted_tokens_list = []
        
        start_time = time.time()
        
        for req_idx, request in enumerate(requests):
            prefix_id = request["prefix_id"]
            token_count = request["token_count"]
            
            # Check cache hit
            if prefix_id in cache:
                cache_hits += 1
                block_id, tokens, _ = cache[prefix_id]
                cache[prefix_id] = (block_id, tokens, time.time())
            else:
                # Cache miss - need to allocate
                if len(cache) >= self.num_blocks:
                    # Evict oldest (LRU)
                    oldest_prefix = min(
                        cache.keys(),
                        key=lambda p: cache[p][2]  # by last_access_time
                    )
                    evicted_tokens = cache[oldest_prefix][1]
                    del cache[oldest_prefix]
                    evictions += 1
                    evicted_tokens_list.append(evicted_tokens)
                
                # Add new block
                cache[prefix_id] = (block_id_counter, token_count, time.time())
                block_id_counter += 1
        
        metrics.cache_hit_rate = cache_hits / len(requests) if requests else 0
        metrics.cache_miss_rate = 1 - metrics.cache_hit_rate
        metrics.total_evictions = evictions
        if evicted_tokens_list:
            metrics.avg_evicted_tokens = np.mean(evicted_tokens_list)
            metrics.total_recompute_tokens = sum(evicted_tokens_list)
        metrics.total_time_seconds = time.time() - start_time
        
        return metrics
    
    def simulate_cost_aware_dual_pool(
        self, 
        requests: List[Dict],
        recency_weight: float = 0.3,
        access_weight: float = 0.4,
        frequency_weight: float = 0.3,
        cost_sensitivity: float = 1.0,
    ) -> SimpleMetrics:
        """
        模擬改進版：Cost-Aware + Dual-Pool。
        """
        metrics = SimpleMetrics(policy_name="Cost-Aware + Dual-Pool (New)")
        
        common_pool = {}  # prefix_id -> (block_id, token_count, access_count, last_access)
        cached_pool = {}  # prefix_id -> (block_id, token_count, access_count, last_access)
        
        max_common_pool_size = self.num_blocks // 5  # 20% for common pool
        block_id_counter = 0
        cache_hits = 0
        evictions = 0
        evicted_tokens_list = []
        
        start_time = time.time()
        
        for req_idx, request in enumerate(requests):
            prefix_id = request["prefix_id"]
            token_count = request["token_count"]
            reuse_probability = request["reuse_probability"]
            
            # Check if in cache (either pool)
            if prefix_id in common_pool:
                cache_hits += 1
                block_id, tokens, access_count, _ = common_pool[prefix_id]
                common_pool[prefix_id] = (block_id, tokens, access_count + 1, time.time())
            elif prefix_id in cached_pool:
                cache_hits += 1
                block_id, tokens, access_count, _ = cached_pool[prefix_id]
                cached_pool[prefix_id] = (block_id, tokens, access_count + 1, time.time())
            else:
                # Cache miss
                total_cached = len(common_pool) + len(cached_pool)
                
                if total_cached >= self.num_blocks:
                    # Evict from cached_pool only (common pool is protected)
                    if cached_pool:
                        # Select eviction candidate by cost-aware score
                        evict_prefix = self._select_eviction_candidate_cost_aware(
                            cached_pool,
                            recency_weight=recency_weight,
                            access_weight=access_weight,
                            frequency_weight=frequency_weight,
                            cost_sensitivity=cost_sensitivity,
                        )
                        
                        evicted_tokens = cached_pool[evict_prefix][1]
                        del cached_pool[evict_prefix]
                        evictions += 1
                        evicted_tokens_list.append(evicted_tokens)
                    else:
                        # cached_pool empty, evict from common_pool as last resort
                        evict_prefix = self._select_eviction_candidate_cost_aware(
                            common_pool,
                            recency_weight=recency_weight,
                            access_weight=access_weight,
                            frequency_weight=frequency_weight,
                            cost_sensitivity=cost_sensitivity,
                        )
                        evicted_tokens = common_pool[evict_prefix][1]
                        del common_pool[evict_prefix]
                        evictions += 1
                        evicted_tokens_list.append(evicted_tokens)
                
                # Decide which pool to add to
                should_common = (
                    reuse_probability > 0.5 and 
                    token_count > 500
                )
                
                if should_common:
                    # Common pool
                    if len(common_pool) >= max_common_pool_size:
                        # Common pool full, move oldest to cached pool
                        oldest_prefix = min(
                            common_pool.keys(),
                            key=lambda p: common_pool[p][3]  # by last_access
                        )
                        moved_data = common_pool.pop(oldest_prefix)
                        cached_pool[oldest_prefix] = moved_data
                    
                    common_pool[prefix_id] = (block_id_counter, token_count, 1, time.time())
                else:
                    # Cached pool
                    cached_pool[prefix_id] = (block_id_counter, token_count, 1, time.time())
                
                block_id_counter += 1
        
        metrics.cache_hit_rate = cache_hits / len(requests) if requests else 0
        metrics.cache_miss_rate = 1 - metrics.cache_hit_rate
        metrics.total_evictions = evictions
        if evicted_tokens_list:
            metrics.avg_evicted_tokens = np.mean(evicted_tokens_list)
            metrics.total_recompute_tokens = sum(evicted_tokens_list)
        metrics.total_time_seconds = time.time() - start_time
        
        return metrics
    
    def _select_eviction_candidate_cost_aware(
        self,
        pool: Dict,
        recency_weight: float,
        access_weight: float,
        frequency_weight: float,
        cost_sensitivity: float,
    ) -> str:
        """
        選擇要驅逐的 block（cost-aware score 最低的）。
        """
        scores = {}
        now = time.time()
        
        for prefix_id, (block_id, token_count, access_count, last_access) in pool.items():
            idle_time = now - last_access
            
            # Component 1: Recency score (idle time normalized)
            recency_score = min(1.0, idle_time / 3600.0)
            
            # Component 2: Access score (lower access = higher eviction priority)
            access_score = 1.0 / (1.0 + access_count)
            
            # Component 3: Cost score (higher token count = LOWER eviction priority)
            cost_score = 1.0 / (1.0 + cost_sensitivity * token_count / 1000.0)
            
            # Combined score
            score = (
                recency_weight * recency_score +
                access_weight * access_score +
                frequency_weight * cost_score
            )
            
            scores[prefix_id] = score
        
        # Return prefix with LOWEST score (highest priority for eviction)
        return min(scores.keys(), key=lambda k: scores[k])
    
    def generate_requests(
        self,
        num_requests: int = 5000,
        workload_type: str = "mixed",
    ) -> List[Dict]:
        """
        生成模擬請求。
        
        每個請求有：
        - prefix_id: 前綴 ID（用於快取鍵）
        - token_count: 該前綴的 token 數量
        - reuse_probability: 下次看到此前綴的概率
        """
        requests = []
        
        if workload_type == "uniform":
            # 所有請求大小相同
            for i in range(num_requests):
                requests.append({
                    "prefix_id": i % 500,  # 500 個不同的前綴
                    "token_count": 512,
                    "reuse_probability": 0.3,
                })
        
        elif workload_type == "long_tail":
            # 少數長前綴，多數短前綴（現實情況）
            for i in range(num_requests):
                if i % 20 == 0:  # 5% 長前綴
                    requests.append({
                        "prefix_id": i % 50,  # 少數不同的長前綴
                        "token_count": np.random.randint(5000, 15000),
                        "reuse_probability": 0.7,  # 高重用率
                    })
                else:  # 95% 短前綴
                    requests.append({
                        "prefix_id": i % 500,
                        "token_count": np.random.randint(100, 1000),
                        "reuse_probability": 0.2,  # 低重用率
                    })
        
        elif workload_type == "mixed":
            # 混合：arxiv 論文、系統提示、普通查詢
            for i in range(num_requests):
                rand = i % 100
                
                if rand < 10:  # 10% arXiv 論文（很長，高重用）
                    requests.append({
                        "prefix_id": i % 30,
                        "token_count": np.random.randint(8000, 20000),
                        "reuse_probability": 0.8,
                    })
                elif rand < 20:  # 10% 系統提示（很短，超高重用）
                    requests.append({
                        "prefix_id": i % 20,
                        "token_count": 100,
                        "reuse_probability": 0.95,
                    })
                else:  # 80% 普通查詢（中等大小，中等重用）
                    requests.append({
                        "prefix_id": i % 600,
                        "token_count": np.random.randint(500, 2000),
                        "reuse_probability": 0.3,
                    })
        
        return requests
    
    def run_comparison(self):
        """
        執行完整對比測試。
        """
        workloads = ["uniform", "long_tail", "mixed"]
        all_results = []
        
        for workload in workloads:
            logger.info(f"\n{'='*80}")
            logger.info(f"Testing workload: {workload}")
            logger.info(f"{'='*80}")
            
            # 生成請求
            requests = self.generate_requests(
                num_requests=5000,
                workload_type=workload,
            )
            
            # 測試 LRU（原版）
            logger.info("Running LRU (Original)...")
            lru_metrics = self.simulate_lru(requests)
            all_results.append(lru_metrics)
            
            # 測試 Cost-Aware + Dual-Pool（新版）
            logger.info("Running Cost-Aware + Dual-Pool (New)...")
            cost_aware_metrics = self.simulate_cost_aware_dual_pool(requests)
            all_results.append(cost_aware_metrics)
            
            # 列印對比結果
            self._print_comparison(lru_metrics, cost_aware_metrics)
        
        return all_results
    
    def _print_comparison(self, lru: SimpleMetrics, cost_aware: SimpleMetrics):
        """
        列印對比結果，計算改進百分比。
        """
        logger.info("\n" + "="*80)
        logger.info("COMPARISON RESULTS")
        logger.info("="*80)
        
        logger.info(f"\n{'Metric':<40} {'LRU':<20} {'Cost-Aware':<20} {'Improvement':<15}")
        logger.info("-"*95)
        
        # Cache Hit Rate
        hit_rate_improvement = (
            (cost_aware.cache_hit_rate - lru.cache_hit_rate) / lru.cache_hit_rate * 100
            if lru.cache_hit_rate > 0 else 0
        )
        logger.info(
            f"{'Cache Hit Rate':<40} "
            f"{lru.cache_hit_rate:.2%}{'':<15} "
            f"{cost_aware.cache_hit_rate:.2%}{'':<15} "
            f"{hit_rate_improvement:+.1f}%"
        )
        
        # Average Evicted Tokens
        evict_improvement = (
            (lru.avg_evicted_tokens - cost_aware.avg_evicted_tokens) / lru.avg_evicted_tokens * 100
            if lru.avg_evicted_tokens > 0 else 0
        )
        logger.info(
            f"{'Avg Evicted Tokens':<40} "
            f"{lru.avg_evicted_tokens:.0f}{'':<15} "
            f"{cost_aware.avg_evicted_tokens:.0f}{'':<15} "
            f"{evict_improvement:+.1f}%"
        )
        
        # Total Recompute Tokens
        recompute_improvement = (
            (lru.total_recompute_tokens - cost_aware.total_recompute_tokens) / lru.total_recompute_tokens * 100
            if lru.total_recompute_tokens > 0 else 0
        )
        logger.info(
            f"{'Total Recompute Tokens':<40} "
            f"{lru.total_recompute_tokens:<20} "
            f"{cost_aware.total_recompute_tokens:<20} "
            f"{recompute_improvement:+.1f}%"
        )
        
        # Total Evictions
        eviction_improvement = (
            (lru.total_evictions - cost_aware.total_evictions) / lru.total_evictions * 100
            if lru.total_evictions > 0 else 0
        )
        logger.info(
            f"{'Total Evictions':<40} "
            f"{lru.total_evictions:<20} "
            f"{cost_aware.total_evictions:<20} "
            f"{eviction_improvement:+.1f}%"
        )
        
        logger.info("-"*95)
        logger.info("\n✓ 正數改進 = 新版本更好")
        logger.info("✗ 負數改進 = 原版本更好\n")


def main():
    """
    主函數：執行基準測試。
    """
    logger.info("Starting Eviction Policy Benchmark...")
    logger.info(f"Timestamp: {datetime.now().isoformat()}")
    
    benchmark = SimpleEvictionBenchmark(num_blocks=1000)
    results = benchmark.run_comparison()
    
    # 導出結果到 JSON
    results_dict = {
        "timestamp": datetime.now().isoformat(),
        "results": [asdict(r) for r in results],
    }
    
    with open("eviction_benchmark_results.json", "w") as f:
        json.dump(results_dict, f, indent=2)
    
    logger.info("\nResults exported to eviction_benchmark_results.json")


if __name__ == "__main__":
    main()
