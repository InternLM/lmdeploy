# chunk gated-delta kernel 测试说明

> 对应测试文件:`tests/pytorch/kernel/test_chunk_gated_delta_rule.py`
> 被测对象:`lmdeploy/pytorch/kernels/cuda/chunk_gated_delta_rule.py`(本地 forward-only Triton 移植,产出 `(o, final_state, chunk_states)`)

本文档逐个说明测试文件中的 test 函数及其作用。全部 40 个用例(含参数化)通过。

---

## 0. 测试辅助(非 test)

- **`_make_inputs(length, ...)`**([L43](../tests/pytorch/kernel/test_chunk_gated_delta_rule.py#L43))
  造 q/k/v/g/beta/state:q/k 已 l2norm、beta 在 sigmoid 区间、g 为负数(模拟 log 衰减)。
- **`_run_fla(q, k, v, g, beta, initial_state, cu_seqlens)`**([L56](../tests/pytorch/kernel/test_chunk_gated_delta_rule.py#L56))
  直接调 `fla.ops.gated_delta_rule.chunk_gated_delta_rule` 取参考值,配置与本地推理路径一致:`scale=K^-0.5`、`use_qk_l2norm_in_kernel=False`、`transpose_state_layout=True`。

---

## 1. autotune 分桶正确性(2 个,纯结构 / 纯 Python)

### 1.1 `test_chunk_count_bucket`([L18](../tests/pytorch/kernel/test_chunk_gated_delta_rule.py#L18))

- **作用**:验证 `_chunk_count_bucket` 的边界映射。
- **参数化**:11 个 chunk 数 → 期望桶值:
  `1/4→4`、`5/16→16`、`17/64→64`、`65/67/128→128`、`129/513→129`。
- **意义**:确保短/中/长序列各落进独立桶,不交叉污染 autotune 配置。
- **特点**:纯 Python,不需要 CUDA。

### 1.2 `test_chunk_kernels_autotune_by_chunk_count_bucket`([L22](../tests/pytorch/kernel/test_chunk_gated_delta_rule.py#L22))

- **作用**:结构回归,逐个检查五个 forward kernel 的 autotuner `keys` 列表里含 `'NT_BUCKET'`。
- **覆盖的 kernel**:
  - `chunk_local_cumsum_scalar_kernel`
  - `chunk_gated_delta_rule_fwd_kkt_solve_kernel`
  - `recompute_w_u_fwd_kernel`
  - `chunk_gated_delta_rule_fwd_kernel_h_blockdim64`
  - `chunk_fwd_kernel_o`
- **意义**:防止后续重构误删该 key 导致"调用顺序污染长序列配置"的 bug 回归。
- **不断言**:具体 best config 或耗时(随 GPU/Triton 版本变化)。

---

## 2. 与 FLA 的数值一致性(4 个,核心正确性)

### 2.1 `test_chunk_gated_delta_rule_matches_fla`([L75](../tests/pytorch/kernel/test_chunk_gated_delta_rule.py#L75))

- **作用**:最基础的对比。连续输入,对比本地 output / final_state 与直接 import 的 FLA 是否一致。
- **参数化**:length ∈ {1, 63, 64, 65, 127, 128, 200} × {有/无 initial_state}(共 14 个)。
- **断言**:
  - output 对上 FLA(`atol=2e-2`);
  - final_state 对上 FLA(`atol=2e-2`);
  - `chunk_states` 形状 == `(1, ⌈T/64⌉, HV, V, K)`;
  - `chunk_states[:, 0]` 逐位等于输入初态(`atol=0`;无初态时等于零)。
- **意义**:chunk 数边界(1/63/64/65/127/128/200)覆盖单 chunk 内、恰对齐、跨边界、多 chunk 多种情况。

### 2.2 `test_chunk_gated_delta_rule_fused_qkv_view_production_shape`([L120](../tests/pytorch/kernel/test_chunk_gated_delta_rule.py#L120))

- **作用**:**fused-QKV 非连续 v 视图**回归。这是移植时踩过的真实 bug——漏了 FLA `input_guard` 的 `.contiguous()`,导致对非连续 v 的指针算术读到错位内存。
- **参数化**:7 个 shape,覆盖:
  - GVA 比例 1:1 与 1:2;
  - K == V 与 K != V(非对称);
  - head dim 64 / 128 / 256(256 为 kernel 支持上限);
  - 长度 63(单 chunk 内)/ 65 / 129(跨 chunk)/ 200(多 chunk)/ 4288(生产 prompt)。
- **断言**(三重):
  1. 非连续 v 视图 == 强制 `.contiguous()` 输入(逐位 `atol=0`)——验证入口规整有效;
  2. output 对上 FLA(`atol=2e-2`);
  3. final_state 对上 FLA(`atol=2e-2`)。
- **意义**:只喂连续输入无法暴露"忘了规整布局"这类回归;生产路径恰走非连续视图。

### 2.3 `test_chunk_gated_delta_rule_packed_varlen`([L155](../tests/pytorch/kernel/test_chunk_gated_delta_rule.py#L155))

- **作用**:**packed 变长多序列**。cu_seqlens = [0, 63, 128, 257](三段 63/65/129)。
- **断言**:
  - output / final_state 对上 FLA;
  - `chunk_states` 的 NT = 6(各序列 chunk 数之和 ⌈63/64⌉+⌈65/64⌉+⌈129/64⌉ = 1+2+3);
  - 每序列起始处的 `chunk_states[0, chunk_offset]` 逐位等于该序列 initial_state(`atol=0`,chunk_offset ∈ {0, 1, 3})。
- **意义**:验证 chunk 偏移在多序列下正确分段,chunk 边界状态按序列对齐。

### 2.4 `test_chunk_gated_delta_rule_tp4_service_shape_with_metadata`([L238](../tests/pytorch/kernel/test_chunk_gated_delta_rule.py#L238))

- **作用**:**生产 TP4 形状端到端**。最贴近线上 prefill 的形态。
- **形状**:H=4, HV=8, K=V=128, T=4288;fused-QKV 非连续 v;带 cu_seqlens + 预计算 `chunk_indices`/`chunk_offsets`。
- **断言**:
  - output / final_state 对上 FLA;
  - `chunk_states.shape == (1, 67, 8, 128, 128)`(67 = ⌈4288/64⌉)。
- **意义**:同时验证 GVA、非连续视图、metadata 复用、多 chunk 边界状态,在生产规模下对齐 FLA。

---

## 3. 集成 / 语义 / 边界(5 个,接线与防御)

### 3.1 `test_chunk_gated_delta_rule_prefill_decode_handoff`([L188](../tests/pytorch/kernel/test_chunk_gated_delta_rule.py#L188))

- **作用**:**prefill(本地 Triton)→ decode(TileLang recurrent)状态交接**。
- **流程**:
  1. 本地 prefill 产出 `final_state`;
  2. 喂给 `fused_recurrent_gated_delta_rule`(TileLang decode kernel)做 3 token decode;
  3. 与"FLA prefill → 同一 decode kernel"逐 token 对比输出和状态。
- **断言**:逐 token output / state 对上(`atol=2e-2`)。
- **意义**:线上 prefill/decode 切换时,本地产出的边界状态与 decode kernel 互操作正确。
- **依赖**:需 tilelang。

### 3.2 `test_chunk_gated_delta_rule_reuses_precomputed_metadata`([L215](../tests/pytorch/kernel/test_chunk_gated_delta_rule.py#L215))

- **作用**:**预计算 metadata vs fallback 自构造等价**。
- **流程**:同一份输入,一次传 `chunk_indices`/`chunk_offsets`、一次不传(让 wrapper 自构造)。
- **断言**:两者的 `(o, final_state, chunk_states)` 三元组逐位相等(`atol=0`)。
- **意义**:验证"metadata 复用"优化不改变结果,且预计算路径与内部 fallback 一致——保证线上一次性构造 metadata 后跨层复用是安全的。

### 3.3 `test_update_chunked_gated_delta_rule_meta_prepares_once`([L267](../tests/pytorch/kernel/test_chunk_gated_delta_rule.py#L267))

- **作用**:**metadata 一次性构造**(TTFT 修复线)。
- **流程**:monkeypatch 计数 `prepare_chunk_indices`/`prepare_chunk_offsets`,调用 `CudaOpsBackend.update_chunked_gated_delta_rule_meta` 一次。
- **断言**:
  - 两个 helper 各只被调用 **1 次**(不是每层一次);
  - 产出的 `gated_delta_chunk_indices.shape == (2, 2)`、`gated_delta_chunk_offsets.shape == (2,)`。
- **意义**:验证 36 个 gated-delta 层共享一份 metadata,而非每层重建——这是消除 TTFT 回退主因之一。

### 3.4 `test_chunk_gated_delta_rule_rejects_invalid_varlen_inputs`([L297](../tests/pytorch/kernel/test_chunk_gated_delta_rule.py#L297))

- **作用**:**输入校验**(host-visible,无 `.item()` D2H 同步)。
- **两处非法**:
  1. cu_seqlens 下物理 batch 必须 = 1(用 `q.expand(2,...)` 制造非法 batch,期望 `ValueError: physical batch size`);
  2. initial_state 行数必须等于序列数(制造行数不匹配,期望 `ValueError: initial_state must have shape`)。
- **意义**:校验只在 host 端做(类型/shape/dtype/rank),不读 CUDA 标量,不破坏跨层 enqueue 重叠。

### 3.5 `test_cuda_backend_updates_only_selected_state_rows`([L314](../tests/pytorch/kernel/test_chunk_gated_delta_rule.py#L314))

- **作用**:**state bank 选择性写回**。
- **流程**:5 行 state_bank、state_indices=[3, 1],跑 `CudaGatedDeltaRuleImpl.chunk_gated_delta_rule`。
- **断言**(四件):
  1. 返回的 bank 与传入 bank 是同一块内存(`data_ptr` 相等,原地更新);
  2. output / chunk_states 与直接调 kernel 的参考值逐位相等(`atol=0`);
  3. 第 1、3 行被正确更新为新状态;
  4. 第 0、2、4 行(未被选中的行)逐位保持不变(`atol=0`)。
- **意义**:验证 state bank 不会污染无关序列的 state——生产中多个序列共用一个 state bank,只能改自己那几行。
- **依赖**:需 tilelang。

---

## 4. 覆盖矩阵总览

| 类别 | test 函数 | 关键验证点 |
|---|---|---|
| 分桶 | `test_chunk_count_bucket` | 桶边界映射正确 |
| 分桶 | `test_chunk_kernels_autotune_by_chunk_count_bucket` | 五个 kernel key 含 `NT_BUCKET` |
| 数值 | `test_chunk_gated_delta_rule_matches_fla` | 基础 length×state 对比 FLA |
| 数值 | `test_chunk_gated_delta_rule_fused_qkv_view_production_shape` | 非连续 v 视图,多 shape |
| 数值 | `test_chunk_gated_delta_rule_packed_varlen` | 多序列 packed varlen |
| 数值 | `test_chunk_gated_delta_rule_tp4_service_shape_with_metadata` | 生产长度端到端 |
| 集成 | `test_chunk_gated_delta_rule_prefill_decode_handoff` | prefill→decode 状态交接 |
| 集成 | `test_chunk_gated_delta_rule_reuses_precomputed_metadata` | 预计算 metadata 与 fallback 等价 |
| 集成 | `test_update_chunked_gated_delta_rule_meta_prepares_once` | metadata 每 step 只构造一次 |
| 边界 | `test_chunk_gated_delta_rule_rejects_invalid_varlen_inputs` | 非法输入校验 |
| 集成 | `test_cuda_backend_updates_only_selected_state_rows` | state bank 选择性写回 |

---

## 5. 容差与断言策略

- **数值对比**(`atol=2e-2, rtol=2e-2`):BF16 下 chunk 累积状态有数值漂移,逐位相等不现实;2e-2 是与 FLA 自身多路径重计算一致的实际可达精度。
- **逐位断言**(`atol=0, rtol=0`):只用在保证逻辑等价、不涉及浮点累积的地方——metadata 等价、state 初值、state bank 未选中行、非连续 vs 连续视图规整后等价。
- **不写入性能阈值**:不在 pytest 中断言具体 best config 或耗时,因为它们随 GPU/Triton 版本变化。

---

## 6. 运行方式

```bash
# 全套
python -m pytest tests/pytorch/kernel/test_chunk_gated_delta_rule.py -q

# 单个 test
python -m pytest tests/pytorch/kernel/test_chunk_gated_delta_rule.py::test_chunk_count_bucket -q

# 参数化子用例
python -m pytest "tests/pytorch/kernel/test_chunk_gated_delta_rule.py::test_chunk_gated_delta_rule_matches_fla[65-True]" -q
```

当前结果:40 passed, 2 warnings。其中 2 个 warning 为无关的 swig `__module__` DeprecationWarning。
