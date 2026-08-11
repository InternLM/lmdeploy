# 本地 chunk gated-delta kernel 移植与调优记录

> 分支:`chunk_cache_kernel`(基于 `main`)。本文档总结从“理解 LMDeploy recurrent state 存储”到“移植 FLA chunk kernel + 产出 chunk 边界状态”再到“修复 TTFT 回退 + autotune 顺序依赖”的全部需求与改动。仅涉及 prefill chunk kernel 与相关元数据/测试;decode kernel、scheduler、block-trie、checkpoint 消费均不在本阶段范围内。

---

## 1. 需求演进

### 1.1 理解框架的 recurrent state 存储

- 定位 linear/gated-delta 的 recurrent state 存储位置与布局。
- 确认 state 更新是否为原地替换。
- 解释 `state_indices` 的作用:把活跃逻辑序列映射到持久 state bank 的行。
- 解释为何 prefill 可见地保存 `recurrent_state` 但不见 `conv_state`。

结论(详见 `lmdeploy/pytorch/paging/state_manager.py` 及 `backends/cuda/gated_delta_rule.py`):
- production V-first 布局为 `[N, HV, V, K]`。
- `state_indices` 把逻辑序列映射到 state bank 行。
- prefill 写 final state 到 bank,conv state 在别处管理,故 prefill 可见只看到 recurrent。

### 1.2 参考 Marconi 设计 chunk 边界 state 存储

参考 `notes/marconi-prefix-caching-for-hybrid-llms.md`,产出“每个 prefill 过程中任意一个 chunk 边界的 state”所需工作的高层 plan。动机:混合模型的 recurrent 层原地更新,前缀缓存需要 chunk 边界处的 state 快照,才能在命中前缀时直接恢复状态、跳过冗余计算。

### 1.3 移植 FLA chunk kernel 到本仓库

- 定位 `/root/flash-linear-attention` 中 `self.chunk_func` 的实现。
- 移植其推理 forward 路径到本仓库,**不调库**,并额外产出 chunk 边界状态。
- 实现语言:用户最初要求 TileLang,后明确“使用 triton 就可以,我只希望不使用 cuda”。本环境同时支持 triton 与 tilelang,但**禁用 native CUDA**。
- 范围限制:“只做 kernel 移植+产出边界状态”。不实现 scheduler、block-trie、prefix-checkpoint 放置/恢复。

### 1.4 正确性与性能验证

- 测试本地 kernel 与直接调 FLA 的数值差异。
- warmup 后多次重复基准,确认性能稳定。
- 排查 decode 是否受影响,以及为何真实 Qwen3-Next 输出完全错误;重新实现/修复。
- 确认线上 Qwen3-Next prefill 用本地 kernel、decode 仍用原 TileLang recurrent kernel。
- 解释短中序列本地快、长序列本地慢的原因。

### 1.5 服务 TTFT 回退排查与修复

- 启动 `lmdeploy serve api_server .../Qwen3-Next-80B-A3B-Instruct/ --server-port 23333 --tp 4 --enable_prefix_caching`,用 `openai_client.py` 请求,新 kernel TTFT 比旧的大。
- 用户明确:`main` 分支是修改前的旧 FLA 基线,`chunk_cache_kernel` 是新分支。
- 修复后确认 kernel 性能结论是否仍成立。

### 1.6 性能差距定位与 autotune 分桶

- 给出本地 kernel 对比 FLA 在不同 sequence length 时的性能差异。
- 排查 inter-chunk recurrent-state propagation 是否为长序列性能差距的根因。
- 发现真实根因是 Triton autotune key 未含 chunk 数,导致短序列配置污染长序列。
- 按用户要求:给受影响 kernel 的 autotune key 加入离散 chunk-count bucket,而非原始 T。

---

## 2. 技术背景

### 2.1 FLA chunk forward 的五个阶段

1. chunk-local 累积 gate cumsum(log2 空间)。
2. intra-chunk KKT + 三角求解。
3. WY 重组(w/u)。
4. inter-chunk recurrent-state 传播(串行)。
5. output 投影。

本地移植的五个对应 Triton kernel:
- `chunk_local_cumsum_scalar_kernel`
- `chunk_gated_delta_rule_fwd_kkt_solve_kernel`
- `recompute_w_u_fwd_kernel`
- `chunk_gated_delta_rule_fwd_kernel_h_blockdim64`
- `chunk_fwd_kernel_o`

### 2.2 生产形状(Qwen3-Next TP4 单卡)

- 48 层;每 4 层 1 个全注意力层;12 个全注意力层 + 36 个 gated-delta 层。
- TP4 单卡 gated-delta 形状:`H=4, HV=8, K=V=128`。
- `chunk_size` 固定 64,与 LMDeploy 64-token KV block 对齐。
- 客户端 prompt 约 4288 token = 67 chunks。
- 默认 `max_prefill_token_num=8192`,4288 不被拆分;单次 prefill forward 的 `T=4288`。

### 2.3 recurrent state 与 chunk 边界状态语义

- `chunk_states[:, 0] = initial_state`(第 0 个 chunk 之前)。
- `chunk_states[:, c]` = 处理完 chunk 0..c-1 后的状态,代表 token 0..c*64。
- `final_state` = 处理完最后一个 chunk 后的状态,代表整段序列。
- `chunk_states[:, -1]` **不是** `final_state`:前者是最后一个 chunk 开始前,后者是最后一个 chunk 处理完。
- V-first 布局 `chunk_states` shape 为 `[B, NT, HV, V, K]`。

---

## 3. 主要改动

### 3.1 新增本地 forward-only Triton kernel

文件:`lmdeploy/pytorch/kernels/cuda/chunk_gated_delta_rule.py`

- 从 `fla/ops/gated_delta_rule/chunk.py` 的 fwd-only 路径移植,无 autograd/backward。
- 固定 `chunk_size=64`,`@triton.jit(do_not_specialize=['T'])` 保持 T 动态。
- 相对 FLA 的额外能力:把 state-forward kernel 的 `h` 作为第三个返回值 `chunk_states` 暴露,供后续 prefix-cache checkpoint 复用。该 `h` 本就是 output kernel 的输入,不新增额外计算。
- 关键 helper:
  - `prepare_lens` / `prepare_chunk_indices` / `prepare_chunk_offsets`(移植自 FLA utils)。
  - `_segmented_arange`。
  - `chunk_local_cumsum_scalar`、`chunk_gated_delta_rule_fwd_intra`、`recompute_w_u_fwd`、`chunk_gated_delta_rule_fwd_h`、`chunk_fwd_o`、`chunk_gated_delta_rule_fwd`(orchestrator)。
  - `_validate_inputs`。
- 复刻 FLA `input_guard` 的必要语义:进入 `q.device` context,对 q/k/v/g/beta/state/cu_seqlens/chunk_indices/chunk_offsets 做 `.contiguous()`。这对 fused-QKV 非连续 v 视图的正确性是必须的。
- 硬件可移植性:
  - capability-aware TF32/IEEE 处理。
  - Blackwell state recurrence 限制为 2 warps,避免四-warp 竞态。
  - shared-memory-aware 的 BV / stage 候选过滤。
- 公开函数签名(最终形态):

```python
def chunk_gated_delta_rule(
    q, k, v, g, beta,
    scale: float | None = None,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = True,
    cu_seqlens: torch.Tensor | None = None,
    state_v_first: bool = True,
    chunk_size: int = 64,
    chunk_indices: torch.Tensor | None = None,
    chunk_offsets: torch.Tensor | None = None,
):  # -> (o, final_state, chunk_states)
```

### 3.2 metadata 一次性构造与复用(TTFT 修复线)

动机:修复前本地 wrapper 在 36 个 gated-delta 层每层重新构造 `chunk_indices`/`chunk_offsets`,且 `repeat_interleave` 路径有潜在同步;被放大 36 次成为 TTFT 回退主因之一。

改动:
- `lmdeploy/pytorch/backends/attention.py`:`AttentionMetadata` 新增
  `gated_delta_chunk_indices` / `gated_delta_chunk_offsets` 两个字段。
- `lmdeploy/pytorch/backends/cuda/op_backend.py`:
  `update_chunked_gated_delta_rule_meta` 在每个非 decode prefill step 调用一次,
  用 `cu_seqlens_q` 构造两个 metadata 并挂到 `attn_metadata`。
  触发点:`update_step_context` 内 `if is_gated_delta and not step_context.is_decoding`。
  移除了原先仅为 FLA tensor cache 服务的预热调用。
- `lmdeploy/pytorch/nn/gated_delta.py`:`GatedDeltaMeta` 引用这两个字段;
  `__call__` 在 prefill 分支把它们传给本地 kernel,decode 分支不构造不使用。
- `lmdeploy/pytorch/backends/gated_delta_rule.py` 与
  `lmdeploy/pytorch/backends/cuda/gated_delta_rule.py`:
  prefill API 增加 `chunk_indices`/`chunk_offsets` 参数并下传;
  state bank 选择/写回(`index_select`/`index_copy_` 或 speculative `_state_select`/`_state_scatter`)不变。
- fallback 保留:直接调 kernel API 且未传 metadata 时,wrapper 仍自行构造,保证专项测试可用。

这两个 metadata 与 FLA 内部 helper 语义一致:
- `chunk_indices [NT,2]` = `[sequence_id, sequence_local_chunk_id]`,把全局 chunk 槽映射到序列内 chunk。
- `chunk_offsets [N+1]` = 各序列 chunk 数前缀和,指示各序列在 `h` 输出张量 `NT` 维的起始位置。

### 3.3 去掉热路径 D2H 同步校验(TTFT 修复线)

移植时误把“FLA 不做的 cu_seqlens 内容校验”当作 input_guard 的一部分加进来:

```python
cu_seqlens[0].item() == 0
cu_seqlens[-1].item() == T
(cu_seqlens[1:] < cu_seqlens[:-1]).any().item()
```

每层 3 个 `.item()` 同步点,36 层 108 个,破坏跨层 enqueue 重叠。FLA `input_guard` 只做 device context 与 `.contiguous()`,不读设备标量。

修复:`_validate_inputs` 只保留 host-visible 检查(tensor 类型、`is_cuda`、device、rank、shape、dtype、`chunk_size=64`、state shape、cu_seqlens/chunk_indices/chunk_offsets 的 rank/dtype/numel)。合法内容作为 scheduler 不变量,不进热路径。对应的“start at 0 / end”内容校验测试被移除。

### 3.4 autotune chunk-count 分桶(顺序依赖修复线)

根因:Triton autotune key 不含 token/chunk 数,同 `H/HV/K/V` 下第一个长度决定后续所有长度复用的配置。
实测:64 先调优时本地 8192 = 0.4305 ms;8192 自调优时 = 0.3455 ms(与 FLA 持平)。分阶段确认短序列配置(KKT 2 warps、state 2 stages)被错误复用于长序列。

改动(仅 `chunk_gated_delta_rule.py`):
- 新增纯 Python helper:

```python
def _chunk_count_bucket(num_chunks: int) -> int:
    if num_chunks <= 4:   return 4
    if num_chunks <= 16:  return 16
    if num_chunks <= 64:  return 64
    if num_chunks <= 128: return 128
    return 129
```

- 五个 forward kernel 均加 `NT_BUCKET: tl.constexpr` 参数,并在 `@triton.autotune(key=[...])` 加入 `'NT_BUCKET'`。
- 各 wrapper 在已有 `NT`(dense=`cdiv(T,64)`,packed=`len(chunk_indices)`)计算后传 `NT_BUCKET=_chunk_count_bucket(NT)`。
- `NT_BUCKET` 仅用于 autotune 缓存区分,**不参与任何数学或地址计算**。
- 保留 `do_not_specialize=['T']`,不按每个原始长度编译。
- 不读 CUDA tensor 内容,无 `.item()`,无 D2H 同步。

bucket 边界设计依据:
- 64/65 分界:4096 token(64 chunks)与生产 4288(67 chunks)分开。
- 128/129 分界:默认 8192 prefill 预算(128 chunks)与超长分开。
- 覆盖现有测试长度(1/63/64/65/127/128/200)、生产 4288、默认 8192、长上下文压力(513 chunks)。

### 3.5 decode 与模型接口保持不变

- decode 仍用 `lmdeploy/pytorch/kernels/cuda/gated_delta_rule.py` 的 TileLang `fused_recurrent_gated_delta_rule`。
- decode 不构造/传递/使用 chunk metadata,`chunk_states=None`。
- 模型(`qwen3_next.py` / `qwen3_5.py`)外部返回契约保持原 tensor-only;`chunk_states` 不向 DecoderLayer/模型外传播,在 GatedDelta/backend 层接收后供 kernel 测试及后续 checkpoint writer 使用(本阶段不接 scheduler)。
- 未修改用户对 Qwen 模型文件的其他变更,未触碰无关文件。

---

## 4. 测试

文件:`tests/pytorch/kernel/test_chunk_gated_delta_rule.py`

现有覆盖:
- 长度 1/63/64/65/127/128/200,有无 initial state,本地 vs FLA 输出/final state 一致性(`atol=2e-2`)。
- fused-QKV 非连续 v 视图(production shape)与连续输入逐位一致 + FLA 一致。
- packed varlen 多序列,chunk_states 的初始状态逐位匹配。
- prefill→TileLang decode 多 token 交接一致性。
- 预计算 metadata 与 fallback 路径逐位等价。
- TP4 生产形状 `T=4288 H=4 HV=8 K=V=128`,67 chunk states。
- `update_chunked_gated_delta_rule_meta` 一次 prefill step 只构造一次 metadata(monkeypatch 计数)。
- state-bank 按 `state_indices` 只更新选中行,未触碰行不变。
- 结构性非法输入校验(physical batch、initial_state shape)。

新增覆盖(分桶):
- `_chunk_count_bucket` 边界参数化:1/4, 5/16, 17/64, 65/67/128, 129/513。
- 五个 autotuned kernel 均含 `NT_BUCKET` key 的结构测试。
- 不在 pytest 中断言具体 best config 或耗时(随 GPU/Triton 版本变化)。

测试结果:34 passed, 2 warnings。

---

## 5. 验证与性能结论

### 5.1 数值一致性

修复 fused-QKV 非连续输入前:output max_abs ≈ 0.2007,state max_abs ≈ 1.8398。
复刻 `input_guard`(device context + `.contiguous()`):output max_abs = 0,state max_abs = 0。

### 5.2 服务 TTFT(控制 A/B,`max_tokens=1`,确认 cache miss)

- main 旧 FLA:warmup 928.81 ms,warm miss median 183.79 ms。
- 修复前本地(50fc5bc4):warmup 942.70 ms,warm miss median 201.94 ms。
- 修复后本地:warmup 925.99 ms,warm miss median 183.90 ms(与 main 差约 0.06%)。
- 所有请求 `cached_tokens=0`、`prefix_cache_hit_rate=0`。

### 5.3 直接 kernel 微基准(NVIDIA H20, BF16, H=4 HV=8 K=V=128, `FLA_INTRACARD_CP=0`)

分桶修复后,调用顺序不再影响结果:

| T (tokens) | 本地 ms | FLA ms | 本地 vs FLA |
|---:|---:|---:|---|
| 64   | 0.1909 | 0.2441 | 快 ~22% |
| 4288 | 0.1939 | 0.2496 | 快 ~22% |
| 8192 | 0.3463 | 0.3465 | 持平 |

修复前对比(顺序依赖):64→8192 时本地 8192 = 0.4305 ms(慢 24%);8192→64 时本地 8192 = 0.3455 ms(持平)。分桶后两种顺序均为 ~0.345 ms。

实测五个 kernel 的 autotune cache 在跨长度触发后均建立独立 bucket entry:`[4, 16, 64, 128, 129]`。

### 5.4 线上 autotune 时序

- 服务无专门 gated-delta 预热;启动 warmup 是 dummy 1-token prefill(`max_batches` 个序列),只触发对应 `max_batches` bucket。
- 第一个真实长 prompt 请求触发其 bucket 的 JIT+autotune(36 层各一遍),该 bucket 内后续复用。
- bucket 128 覆盖 4288 与 8192;短请求走自己的 bucket,不再污染长序列配置。
- 若显式把 `max_prefill_token_num` 调小到 4096,4288 被拆:T=4096(64 chunks,bucket 64)+ T=192(3 chunks,bucket 4)。

### 5.5 关于 inter-chunk recurrence 的修正

早期结论“8192 慢是 inter-chunk recurrence 较弱”证据不足。对照显示本地与 FLA common Triton 路径的 launch grid、串行循环、FP32 accumulator、per-chunk state store、指针/int64 offset、H20 autotune 候选基本一致;`FLA_INTRACARD_CP=0` 下两边都用串行 recurrence。真正原因是 autotune key 缺 chunk 数导致的配置复用,而非 recurrence 算法差异。

---

## 6. 已知边界与后续方向

- checkpoint 消费未接:scheduler/block-trie/state checkpoint 分配与恢复未实现(本阶段明确排除)。
- chunk_states 当前在模型层接收后丢弃,等待 checkpoint writer。
- 冷启动:第一个真实新 bucket 请求仍付 autotune 成本;若敏感可在 warmup 增加针对生产长度的 dummy prefill。
- packed varlen 的 bucket 基于 `len(chunk_indices)` 总 chunk 数,不是单序列最大 chunk 数;对长度分布差异大的 packed 批次是近似代理,精确化需同步/归约,本阶段不做。
- 长序列(>8192)优化方向:可参考 FLA intra-card CP(分段并行扫描+合并)移植进本地 kernel,但需处理其 stream 同步问题;当前环境该路径默认关闭、FlashQLA 未安装。
- `state_manager.py` 中有一处早期遗留 debug print,与本改动无关,未触碰。

---

## 7. 涉及文件

改动:
- `lmdeploy/pytorch/kernels/cuda/chunk_gated_delta_rule.py`(新增,主文件)
- `lmdeploy/pytorch/backends/attention.py`(metadata 字段)
- `lmdeploy/pytorch/backends/cuda/op_backend.py`(metadata 一次性构造)
- `lmdeploy/pytorch/backends/gated_delta_rule.py`(抽象 API 扩展)
- `lmdeploy/pytorch/backends/cuda/gated_delta_rule.py`(prefill 传 metadata + 状态写回)
- `lmdeploy/pytorch/nn/gated_delta.py`(GatedDeltaMeta/`__call__`)
- `tests/pytorch/kernel/test_chunk_gated_delta_rule.py`(专项测试)

未改动(刻意保留):
- `lmdeploy/pytorch/kernels/cuda/gated_delta_rule.py`(TileLang decode kernel)
- `lmdeploy/pytorch/models/qwen3_next.py`、`lmdeploy/pytorch/models/qwen3_5.py`(模型接口)
- `lmdeploy/pytorch/paging/state_manager.py`、`openai_client.py` 及其他无关/用户自有文件
