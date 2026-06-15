# Prefix Cache Eviction / Admission：最新實驗設計（給 Claude Code）

## 0. 研究目標

這份文件描述一個 **Phase 1** 實驗版本：

> 不先改 scheduler，也不綁定 Jenga 的 dual pool / common page pool。  
> 先只改 prefix KV cache 的 **admission** 和 **eviction**。

目標是回答：

1. 一個 inactive KV block 值不值得留下來等未來 prefix reuse？
2. GPU memory 不夠時，應該 evict 哪些 inactive KV blocks？
3. 是否可以比 vLLM 的 LRU-style eviction 更好？
4. 是否可以比 Jenga 的 common page pool 更不依賴固定 common-prefix predictor？

Jenga 的背景動機是：保留 inactive tokens 可以提高 prefix cache hit rate，但會壓縮 active tokens 的 memory，進而降低 batch size。因此 prefix caching 本質上是 **reuse benefit** 和 **memory cost** 的 trade-off。

---

## 1. 核心設計摘要

使用 single-pool value-based policy：

```text
所有 inactive cached KV blocks 放在同一個 pool。
每個 block 有一個 value density D_b。
Admission 時看 D_b 是否大於 memory price lambda。
Eviction 時從候選 blocks 中 evict D_b 最低者。
```

不要做：

```text
dual pool
common pool
protected pool
global min-heap 第一版
每次 memory pressure 改變就重排所有 blocks
```

第一版建議保留 vLLM 原本 free queue / linked list，eviction 時只 sample 小量候選 blocks。

---

## 2. 定義

### 2.1 Request

一個 request \(r\) 有：

- prompt tokens
- output tokens
- arrival time
- prefix hash sequence
- optional metadata，例如 session id、template id、agent id、tool id

---

### 2.2 KV block

一個 KV block \(b\) 有：

- block id
- block hash / prefix hash
- layer type
- prefix depth
- memory size \(m_b\)
- estimated reuse probability \(\hat p_b(t,H)\)
- estimated saved prefill cost \(\Delta C_b(t)\)
- value density \(D_b(t)\)

---

### 2.3 Inactive block

一個 block 當前不被 running request 使用，且：

```text
ref_cnt == 0
```

但它仍可能留在 prefix cache 中，供未來 request prefix reuse。

---

### 2.4 Reuse event

如果未來 request 的 prefix match 到 block \(b\)，且 block \(b\) 還在 GPU prefix cache 裡，稱為 reuse / hit。

在 horizon \(H\) 內 hit：

\[
y_b = 1
\]

在 horizon \(H\) 內沒有 hit：

\[
y_b = 0
\]

---

## 3. 為什麼不用完整 \(V_b\) 當 heap key？

原本的 block 保留價值可以寫成：

\[
V_b(t)
=
\hat p_b(t,H)\Delta C_b(t)
-
\lambda_t m_b
\]

其中：

- \(\hat p_b(t,H)\)：block \(b\) 在未來 horizon \(H\) 內被 reuse 的機率
- \(\Delta C_b(t)\)：如果被 reuse，可以省下的 prefill cost
- \(m_b\)：block 佔用的 GPU memory
- \(\lambda_t\)：目前每單位 memory 的價格

但是 \(\lambda_t\) 是全域 memory pressure。它可能每個 scheduler step 都變。

如果 heap key 是完整 \(V_b(t)\)，那每次 \(\lambda_t\) 改變，所有 blocks 的 heap key 都會改變。這會造成：

\[
O(N \log N)
\]

的全量更新，太貴。

---

## 4. 改用 value density \(D_b\)

Admission 條件：

\[
V_b(t) > 0
\]

代入：

\[
\hat p_b(t,H)\Delta C_b(t) - \lambda_t m_b > 0
\]

移項：

\[
\hat p_b(t,H)\Delta C_b(t) > \lambda_t m_b
\]

兩邊除以 \(m_b\)：

\[
\frac{\hat p_b(t,H)\Delta C_b(t)}{m_b} > \lambda_t
\]

因此定義：

\[
D_b(t)
=
\frac{\hat p_b(t,H)\Delta C_b(t)}{m_b}
\]

這裡 \(D_b\) 代表：

> 每單位 GPU memory 可以帶來多少 expected saved prefill cost。

注意：這裡的 \(D_b\) 不是 decibel，而是 density。

---

## 5. Admission rule

當一個 block 變成 inactive 時，計算：

\[
D_b(t)
=
\frac{\hat p_b(t,H)\Delta C_b(t)}{m_b}
\]

如果：

\[
D_b(t) > \lambda_t
\]

則保留這個 block：

```text
admit b into prefix cache
```

否則：

```text
release b or give it lowest cache priority
```

---

## 6. Eviction rule

當需要釋放 GPU memory 時，不掃完整 linked list。

使用 sampling eviction：

```text
從 free queue / evictable inactive blocks 中取 K 個候選。
對候選 blocks 現場計算 D_b。
evict D_b 最低的 block。
```

公式：

\[
b^*
=
\arg\min_{b \in S_K} D_b(t)
\]

其中 \(S_K\) 是本次 sample 到的 \(K\) 個候選 blocks。

---

## 7. Memory price \(\lambda_t\)

\(\lambda_t\) 是目前 GPU memory 的價格。

### 7.1 不要把 \(\lambda_t\) 放進 heap key

\(\lambda_t\) 只用於 admission threshold：

\[
D_b(t) > \lambda_t
\]

不要用：

\[
V_b(t) = D_b(t) - \lambda_t
\]

當 heap key。

這樣 \(\lambda_t\) 改變時，不需要更新所有 cached blocks。

---

### 7.2 第一版建議用 eviction pressure 更新

定義：

\[
e_t
=
\frac{
\text{cached blocks evicted in this scheduler step}
}{
\max(1,\text{blocks allocated in this scheduler step})
}
\]

白話：

> 這個 scheduler step 裡，有多少 allocation 是靠 evict cached blocks 完成的。

更新：

\[
\lambda_{t+1}
=
\max
\left(
0,
\lambda_t
+
\eta(e_t - e_{\text{target}})
\right)
\]

其中：

- \(\eta\)：更新速度
- \(e_{\text{target}}\)：可接受的 cached block eviction ratio

建議初始值：

```text
eta = 0.01
e_target = 0.05
lambda_0 = small positive value or 0
```

直覺：

```text
如果最近常常需要 evict cached blocks：
    lambda 上升，admission 更嚴格

如果最近很少 evict cached blocks：
    lambda 下降，admission 更寬鬆
```

---

## 8. Reuse probability \(\hat p_b(t,H)\)

目標是估：

\[
\hat p_b(t,H)
=
P(\text{block } b \text{ 在未來 } H \text{ 內會被 reuse})
\]

第一版使用：

```text
分群統計法 + workload-aware prior
```

---

### 8.1 Horizon \(H\)

先定義 reuse horizon。

建議做兩種：

```text
H_time = 30 seconds
H_req = 500 requests
```

第一版可只用其中一種，例如：

```text
H = 500 future requests
```

如果 block 在 horizon 內被 hit：

\[
y = 1
\]

如果 horizon 到期沒 hit：

\[
y = 0
\]

---

### 8.2 Block class

不要對每個 prefix hash 單獨統計，太 sparse。

定義 block class：

\[
k(b)
=
(
\text{layer type},
\text{prefix length bucket},
\text{workload type}
)
\]

第一版可以先簡化為：

\[
k(b)
=
(
\text{layer type},
\text{prefix length bucket}
)
\]

prefix length bucket 例如：

```text
0-512
512-2K
2K-8K
8K+
```

---

### 8.3 分群統計法

對每個 class \(k\)，維護：

- \(A_k\)：admitted blocks 在 horizon 內 hit 的次數
- \(B_k\)：admitted blocks 在 horizon 內沒有 hit 的次數

估計：

\[
\hat p_k
=
\frac{A_k + a_0}{A_k + B_k + a_0 + b_0}
\]

建議：

```text
a0 = 1
b0 = 9
```

代表一開始保守假設 reuse probability 約 10%。

更新：

```text
如果 block hit within H:
    A_k += 1

如果 block expires without hit:
    B_k += 1
```

---

### 8.4 Decay

為了適應 workload 變化，定期衰減舊統計：

\[
A_k \leftarrow \rho A_k
\]

\[
B_k \leftarrow \rho B_k
\]

建議：

```text
rho = 0.95
decay every 1000 requests or every 60 seconds
```

---

## 9. Workload-aware prior

可以根據當前 workload 狀態調整 reuse probability。

這不是手動說：

```text
if agent then p *= 2
```

而是估計 workload mode distribution：

\[
q_t(z)
\]

其中 \(z\) 可以是：

```text
agent / tool-use
multi-turn chat
RAG / long-document QA
batch benchmark
random chat
code generation
```

對每種 workload mode \(z\)，維護 reuse prior：

\[
\mu_z
=
\frac{A_z + a_0}{A_z + B_z + a_0 + b_0}
\]

目前 workload prior：

\[
p_{\text{prior}}(t)
=
\sum_z q_t(z)\mu_z
\]

再和 block class 的統計合併：

\[
\hat p_b(t,H)
=
\frac{
n_k \hat p_k
+
\kappa p_{\text{prior}}(t)
}{
n_k+\kappa
}
\]

其中：

\[
n_k = A_k + B_k
\]

\(\kappa\) 控制 workload prior 的影響力。

建議：

```text
kappa = 10
```

直覺：

```text
如果 class k 的歷史資料很少：
    多相信 workload prior

如果 class k 的歷史資料很多：
    多相信 class-level statistics
```

---

## 10. Workload detection signals

第一版不要用複雜模型，用 sliding window statistics。

對最近 \(W\) 個 requests：

```text
W = 64 or 128
```

計算以下 signals。

### 10.1 Template repeat rate

\[
\text{template\_repeat\_rate}
=
\frac{
\text{most common template hash count}
}{
W
}
\]

高代表 agent / RAG / benchmark 可能性較高。

---

### 10.2 Shared prefix rate

\[
\text{shared\_prefix\_rate}
=
\frac{
\#\{(i,j): \text{LCP}(i,j) > L_0\}
}{
\#\text{sampled pairs}
}
\]

建議：

```text
L0 = 512 tokens
```

高代表 prefix reuse 機會高。

---

### 10.3 Session continuity

\[
\text{session\_repeat\_rate}
=
\frac{
\text{requests whose session id appeared before}
}{
W
}
\]

高代表 multi-turn chat 或 agent。

---

### 10.4 Tool / agent metadata

如果 request metadata 有：

```text
agent_id
tool_call
workflow_id
template_id
conversation_id
```

可以直接提高對 agent / tool-use mode 的 belief。

---

## 11. Workload mode estimation

第一版可以 rule-based，不需要 classifier。

Example：

```text
if tool_call_rate > 0.3 or agent_id_rate > 0.3:
    q(agent) = 0.8
elif shared_prefix_rate > 0.3 and template_repeat_rate > 0.3:
    q(RAG_or_batch) = 0.7
elif session_repeat_rate > 0.3:
    q(multi_turn_chat) = 0.7
else:
    q(random_chat) = 0.7
```

剩餘機率平均分給其他 modes。

注意：agent workload 不是所有 blocks 都高 reuse。通常是以下 blocks 高 reuse：

```text
system prompt
tool schema
agent instruction
workflow prompt
template prefix
```

user-specific input / tool output 不一定高 reuse。

因此 workload prior 只應該作為 prior，不應該覆蓋 block class statistics。

---

## 12. Saved prefill cost \(\Delta C_b(t)\)

第一版可以近似：

\[
\Delta C_b(t)
=
\text{num tokens represented by block}
\times
\text{prefill cost per token}
\times
\text{layer weight}
\]

如果沒有 layer-level profiler，可以先用：

\[
\Delta C_b(t)
=
\text{num tokens represented by block}
\]

也就是用 saved tokens 作為 saved cost proxy。

對不同 layer type 可以加權：

```text
full attention: 1.0
sliding window: lower or based on active_pages relevance
mamba/state: based on recomputation cost
vision embedding: high if recomputing vision encoder is expensive
```

第一版建議先保持簡單：

```text
DeltaC_b = number of prefix tokens saved by this block
```

---

## 13. Sampling K：三階段設計

本設計不建議第一版直接使用 global min-heap。

原因：

1. block score 會因 reuse probability / workload mode 改變而 stale。
2. \(\lambda_t\) 是全域 memory pressure，不應該導致全 heap 更新。
3. vLLM 原本 linked list 支援 O(1) remove/touch，直接換 heap 會破壞這個優點。
4. sampling eviction 可以先驗證 policy 是否有效，工程風險最低。

因此使用三階段 sample \(K\) 設計。

---

## 13.1 Stage 1：固定 K

第一版最小實作：

```text
K = 16
```

Eviction：

```text
need to evict one cached block
    take first K evictable cached blocks from free queue
    compute D_b for each candidate
    evict candidate with lowest D_b
```

如果一次要 evict \(F\) 個 blocks：

```text
take S candidates
evict F candidates with lowest D_b
```

Stage 1 的簡化版本可以先對每個 block 都重複執行 K-sampling。

---

## 13.2 Stage 2：K sensitivity test

實驗時要記錄不同 K 的效果：

```text
K in {1, 4, 8, 16, 32, 64}
```

其中：

```text
K = 1
```

接近原本 linked-list head eviction。

比較指標：

- throughput
- TTFT
- TPOT
- prefix hit rate
- saved prefill tokens
- eviction regret
- CPU scheduling overhead
- number of candidates scored per second

目標是找：

> 效果已經接近飽和，但 overhead 還小的最小 K。

預期：

```text
K = 16 or 32
```

可能是合理區間。

---

## 13.3 Stage 3：adaptive K

如果 Stage 2 顯示 K 對結果敏感，再做 adaptive K。

定義 eviction pressure：

\[
e_t
=
\frac{
\text{cached blocks evicted in this scheduler step}
}{
\max(1,\text{blocks allocated in this scheduler step})
}
\]

使用：

\[
K_t =
\begin{cases}
8, & e_t < 0.05 \\
16, & 0.05 \le e_t < 0.20 \\
32, & e_t \ge 0.20
\end{cases}
\]

白話：

```text
memory pressure low:
    K = 8

memory pressure medium:
    K = 16

memory pressure high:
    K = 32
```

這樣避免所有情況都付 \(K=32\) 的成本。

---

## 13.4 Batch eviction 的 sample size

如果一次需要 evict \(F\) 個 cached blocks，不要每個 victim 都重新 sample K 個。

建議：

\[
S
=
\min
(
N_{\text{evictable}},
\max(16, rF),
64
)
\]

其中：

```text
F = number of blocks to evict
r = 4
S = total sampled candidates
```

然後：

```text
sample S candidates
compute D_b for each candidate
evict lowest F candidates
```

Examples：

```text
F = 4:
    S = max(16, 4*4) = 16

F = 20:
    S = min(64, 4*20) = 64
```

---

## 14. 為什麼 K=16 是合理起點？

假設 bottom 10% blocks 是最想 evict 的低價值 blocks。

看 \(K\) 個候選時，至少看到一個 bottom 10% block 的機率：

\[
1 - (1 - 0.1)^K
\]

大約：

| K | probability |
|---:|---:|
| 8 | 57% |
| 16 | 81% |
| 32 | 97% |
| 64 | 99.9% |

如果 target 是 bottom 5%：

\[
1 - (1 - 0.05)^K
\]

| K | probability |
|---:|---:|
| 8 | 34% |
| 16 | 56% |
| 32 | 81% |
| 64 | 96% |

所以：

```text
K = 16
```

是便宜且有明顯改善機會的起點。

```text
K = 32
```

通常更穩，但 overhead 較高。

---

## 15. Candidate selection

第一版 candidate 可以取 free queue 前 K 個 evictable cached blocks。

但要注意區分：

```text
uncached free blocks
cached free blocks
```

uncached free blocks 不需要 eviction，應該優先用。

candidate 只考慮：

```text
ref_cnt == 0
block_hash is not None
is cached prefix block
```

如果 free queue 前面有 uncached blocks：

```text
直接 allocate，不需要 evict cached block
```

只有當 allocation 需要回收 cached blocks 時，才啟動 sampling eviction。

---

## 16. Ghost / evicted record table

為了知道 evict 錯了沒有，維護 metadata-only table。

當 block 被 evict：

```text
store prefix hash
store layer type
store prefix depth
store evict time
store class k
store estimated DeltaC
```

如果未來 request 需要這個 prefix，但 block 已經被 evict：

```text
ghost hit
```

定義 eviction regret：

\[
\text{EvictionRegret}
=
\sum_{\text{ghost hit } b}
\Delta C_b
\]

這個 metric 很重要，因為它衡量：

> 我們丟掉了多少本來可以 reuse 的 prefix blocks。

---

## 17. Pseudocode

```text
OnBlockInactive(b):
    k = classify_block(b)

    p_class = estimate_class_reuse_probability(k)
    p_prior = estimate_workload_prior()
    p = combine(p_class, p_prior)

    deltaC = estimate_saved_prefill_cost(b)
    D = p * deltaC / size(b)

    b.cached_density = D
    b.class_id = k

    if D > lambda:
        admit b into prefix cache
    else:
        release b or put with lowest priority


OnMemoryPressure(need_blocks F):
    update_lambda()

    if stage == fixed_K:
        S = K
    elif stage == adaptive_K:
        S = choose_K_by_eviction_pressure()
    elif batch_mode:
        S = min(N_evictable, max(16, 4 * F), 64)

    candidates = take_S_evictable_cached_blocks_from_free_queue(S)

    for b in candidates:
        D_b = recompute_density_if_needed(b)

    victims = lowest_F_by_density(candidates)

    evict victims
    insert victim metadata into ghost table


OnCacheHit(b):
    k = b.class_id
    A_k += 1
    update block / class statistics


OnTTLExpireWithoutHit(b):
    k = b.class_id
    B_k += 1


OnGhostHit(meta):
    record eviction regret += meta.deltaC
    update corresponding class as positive reuse signal
```

---

## 18. Baselines

Compare against:

1. vLLM default LRU-style prefix cache eviction
2. Jenga customized LRU if available
3. Jenga common page pool if available
4. Your policy with K=1
5. Your policy with K=16
6. Your policy with K=32
7. Your policy with adaptive K

---

## 19. Metrics

Must record:

```text
throughput
TTFT p50 / p95 / p99
TPOT p50 / p95 / p99
prefix cache hit rate
saved prefill tokens
saved prefill FLOPs if available
eviction regret
ghost hit rate
cache pollution rate
CPU scheduling overhead
number of candidate blocks scored
number of cached blocks evicted
lambda over time
K over time
```

Cache pollution rate:

\[
\text{PollutionRate}
=
\frac{
\text{admitted blocks that expire without hit}
}{
\text{all admitted blocks}
}
\]

Ghost hit rate:

\[
\text{GhostHitRate}
=
\frac{
\text{ghost hits}
}{
\text{ghost table entries}
}
\]

---

## 20. Workloads

Prefer real existing workloads / public datasets.

Suggested workloads:

1. **Long-document QA**
   - QASPER
   - arXiv-QA / arxiv-march-2023 style datasets
   - LongBench

2. **Chat / multi-turn**
   - LMSYS-Chat-1M
   - WildChat-1M

3. **Code generation**
   - HumanEval
   - MBPP
   - APPS if available

4. **Agent / tool-use**
   - τ-bench
   - AgentBench
   - Berkeley Function Calling Leaderboard data if accessible

5. **Benchmark / repeated template**
   - MMLU-Pro
   - GSM8K
   - GPQA if accessible

Do not generate synthetic prompts as the main result.

Synthetic workloads can be used only for ablation, e.g.:

```text
controlled prefix overlap ratio
controlled request arrival burstiness
controlled output length
```

---

## 21. Implementation notes for vLLM

First version should avoid large data structure changes.

Recommended:

```text
keep existing linked list / free queue
add sampling eviction path
compute density only for sampled candidates
do not globally update all cached blocks
do not implement global min-heap in first version
```

Later versions can test:

```text
per-class lazy min-heap
bucket queue
indexed heap
```

But Phase 1 should first test whether value-based eviction is useful.

---

## 22. Expected contribution

The paper contribution should not be phrased as:

> We propose a new score.

Better phrasing:

> We formulate prefix KV cache admission and eviction as an expected-value retention problem. Each inactive block is retained only when its expected saved prefill cost per unit memory exceeds the current memory price. We estimate reuse probability from online feedback and workload signals, and implement a low-overhead sampled eviction policy in vLLM.

Chinese version:

> 我們把 prefix KV cache 的 admission / eviction 形式化為 expected-value retention problem。每個 inactive block 是否保留，取決於它每單位 GPU memory 的預期 prefill savings 是否超過當前 memory price。我們用線上 feedback 和 workload signals 估計 reuse probability，並在 vLLM 中實作低 overhead 的 sampled eviction policy。
