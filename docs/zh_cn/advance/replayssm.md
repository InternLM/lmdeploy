# Gated DeltaNet 与 KDA ReplaySSM

ReplaySSM 是一种推理优化：缓存近期状态空间模型（SSM）的**输入**，并从较早的
checkpoint 重放这些输入，而不是每生成一个 token 都把完整循环状态写回 HBM。
该方法来自 [ReplaySSM 原理文章][replayssm-blog]及其
[参考实现][replayssm-code]。LMDeploy 将其用于 Qwen3-Next 和 Qwen3.5 系列模型
中的 [Gated DeltaNet][gdn-paper] 层，以及 GLM-5.3 Flash 的通道级 KDA 层。

## 当前状态

ReplaySSM 是实验性的 PyTorch CUDA 功能，需要显式开启，默认关闭。受支持的
标准解码、Qwen3.5 的 `qwen3_5_mtp` 和 GLM-5.3 Flash 的 `deepseek_mtp` 配置可以设置 `enable_replayssm=True` 启用。
executor 构造阶段会检查 backend、engine role、外部 cache connector、ring geometry、speculative method、
GPU 架构和模型类型；不受支持的配置会提前报错。当 `enable_replayssm=False` 时，
executor 启动和现有基线路径均不受影响。

当前验证范围仅包括：

- SM90 GPU 上的 PyTorch CUDA backend；
- 同机 `Hybrid` engine；
- Qwen3-Next 和 Qwen3.5 系列 Gated DeltaNet 模型，以及 GLM-5.3 Flash KDA；
- 标准自回归解码或对应模型的 MTP 投机解码。

本检查点尚不支持 TurboMind、非 CUDA 设备、P/D 解耦状态传输、外部 cache
connector 或其他循环架构。

DeepSeek-V4 不含 GDN/KDA 层，**不支持 ReplaySSM**，使用 `deepseek_mtp` 也不例外。
它的 compressor/window 状态属于另一种协议。flag-off 兼容性测试不能作为 ReplaySSM
支持或加速证据。排除其支持范围不等于证明 flag-off 精度对齐；历史单启动 GSM8K
下降0.834个百分点的原因仍未确定。

启用的 `kv_transfer_config` 会在 executor 分配模型和缓存前被拒绝，包括 producer、consumer、
双向角色及自定义 connector。仅更换 cache key 前缀不能让外部恢复安全：Replay 的
checkpoint/rings/cursor 协议尚未获得传输资格。本地 prefix cache 与外部 connector 不同。

### 性能与验收边界

状态正确性、任务精度、同 batch 延迟、同显存预算容量应分别验收，不能相互替代。
GLM KDA 路径不是通用的同 batch 延迟优化：H200 有状态 ring-cycle 测量中，B128
标准解码从77.41增至99.81微秒，MTP5合成混合接受从280.12增至311.57微秒。
测量包含递推和 cursor 推进，但不是整模延迟或真实生产接受分布。

GLM MTP 已测容量吞吐提升9.50%比较的是固定 KV blocks 下 native B224 与 Replay B512；
Qwen MTP B192 对 B512 则提升0.93%。两者都不证明达到20%收益或同 batch 尾延迟下降。
这些更大性能 batch 的自然 EOS 精度、流式 TTFT/TPOT 尚未验收。已有同 batch 全量 GSM8K
重复均值通过的是约定的单侧下降0.5个百分点门，不是统计非劣性或逐 token 等价保证。
保持默认关闭，应先按目标负载验证再开启。

## GLM-5.3 Flash

引擎同样使用 `enable_replayssm=True` 和 `replayssm_buffer_len=16`。
投机解码使用 `SpeculativeConfig(method='deepseek_mtp', num_speculative_tokens=5)`，
ring 长度必须满足 `L >= 2 * (num_speculative_tokens + 1)`。

KDA 的衰减按 key 通道变化，因此 gate ring 为 `[H,L,K]`，而不是 `[HV,L]`。
AR 和 MTP 的 checkpoint、变换后 delta/key/gate ring 均为 FP32；MTP 额外保留
BF16 原始 key/value ring 与 FP32 beta。重建时衰减作用在 key 因子上。
卷积、chunked prefill、KPool tail 与仅含 MLA 的 MTP predictor 保持原实现。
共享 cursor 仍由 target 每步统一推进，MTP 只提交接受的输入状态转换。

这是实验性支持，不保证 bitwise 一致或一定加速。AR 仍保留完整 checkpoint 并新增
ring；MTP 则替换每个投机位置的完整 KDA 快照。应针对实际负载测量精度、延迟和内存。
下文标量 GDN 的布局及数值策略不变。

## 基本原理

### 基线循环解码

Gated DeltaNet 层维护一个较大的循环矩阵 `S`。简化后，每个 token 会先做衰减，
计算 delta-rule 修正，更新矩阵，然后读取输出：

```text
S_decay = exp(g) * S
d       = beta * (v - S_decay * k)
S_next  = S_decay + d * k^T
y       = S_next * q
```

精确公式参见 [Gated DeltaNet 论文][gdn-paper]。传统解码中，每层、每个 token
都要读写完整的 `S[HV,V,K]` 矩阵。虽然一个 token 新增的信息只有紧凑向量
`(d,k,g)`，完整状态流量仍可能主导单 token 解码开销。

### 标准 ReplaySSM

标准 ReplaySSM 将逻辑循环状态表示为：

```text
逻辑状态 = checkpoint S0 + 已提交的 (d,k,g) ring 前缀 + cursor
```

普通步骤只把 `(d,k,g)` 追加到长度为 `L` 的环形缓存，并从 `S0` 和已缓存历史
直接计算输出，不物化和写回新的完整状态。当 ring 写满，或者 cache 生命周期要求
安全边界时，再把缓存的状态转移一次性折叠到 `S0` 并清空 ring。因此完整状态写回
被摊销到 `L` 个解码步骤。

它主要优化显存流量，而不是状态容量：标准 ReplaySSM 仍保留一份完整 checkpoint，
并额外增加 ring。在固定 GPU 显存预算下，额外 ring 可能减少可用 state-cache
slot 数量。

### MTP ReplaySSM

传统 Gated DeltaNet 投机验证可能为每个 verify 位置保留一份完整循环状态快照，
以便拒绝 token 后恢复最后一个已接受位置。当
`Tmax = num_speculative_tokens + 1` 时，这部分存储会随 `Tmax` 增长。

MTP ReplaySSM 改为保存一份 FP32 checkpoint、紧凑的变换后/原始输入 ring 和
device cursor。候选记录先保持 tentative；rejection sampling 完成后，cursor 只按
已接受的 target-input 数量前进。被拒绝的后缀记录不可见，之后会被覆盖。需要
flush 时，LMDeploy 按原顺序重放已经提交的 raw inputs，从而精确推进 checkpoint。
这样即可移除逐位置完整状态快照，并把 rollback 变成 cursor 操作。ring/cursor
设计及其生命周期约束参考了原始 ReplaySSM 工作。

ReplaySSM 不改变模型权重、tokenizer、采样算法或 attention KV cache 语义。但
输出重构可能采用不同的浮点累加顺序，因此必须实际验证 accuracy 和 MTP
acceptance，不能假设结果 bitwise 相同。

### 缓存内容与 shape

下面以一个 request slot、一个 Gated DeltaNet 层为单位定义：

- `H`：当前 TP rank 上的 query/key head 数；
- `HV`：当前 TP rank 上的 value head 数；
- `K`、`V`：query/key 与 value 的 head dimension；
- `L`：ReplaySSM ring 长度；
- MTP verification 中 `Tmax = num_speculative_tokens + 1`。

每个 value head 拥有一份 `S[HV,V,K]`；每组 `HV/H` 个 value head 共享一个
归一化后的 key/query head。下表是每 slot、每 layer 的逻辑 shape，实际 state
cache allocation 还会带 slot 和 layer 维。packed cursor 例外：每个 slot 只有一份
共享于全部 GDN layer 的 `[3]` 记录。

Standard ReplaySSM 缓存：

| 内容            |      Shape | Dtype           | 含义                                      |
| --------------- | ---------: | --------------- | ----------------------------------------- |
| checkpoint `S0` | `[HV,V,K]` | SSM-state dtype | 上一个 flush boundary 的状态              |
| `d_ring`        | `[HV,L,V]` | SSM-state dtype | delta-rule correction 向量 `d_i`          |
| `k_ring`        |  `[H,L,K]` | SSM-state dtype | value-head group 共享的 L2-normalized key |
| `g_ring`        |   `[HV,L]` | FP32            | log-decay，`alpha_i = exp(g_i)`           |
| cursor          |      `[3]` | int32           | `[write_pos, cache_base, is_flush]`       |

普通 standard 路径中 `cache_base` 和 `is_flush` 为零。non-flush token 只追加
`(d_i,k_i,g_i)`；不缓存 `q_i`、`v_i`、`beta_i`。原因是 `q_i` 只读取状态，
而 `v_i/beta_i` 已经被吸收到 `d_i` 中。

MTP ReplaySSM 还会缓存 raw inputs，以便使用原始 closed-loop recurrence 折叠
persistent checkpoint：

| 内容            |      Shape | Dtype            | 含义                                                  |
| --------------- | ---------: | ---------------- | ----------------------------------------------------- |
| checkpoint `S0` | `[HV,V,K]` | FP32             | 最近一次物化的 committed state                        |
| `d_ring`        | `[HV,L,V]` | activation dtype | 快速重构使用的 correction vector                      |
| `k_ring`        |  `[H,L,K]` | activation dtype | 快速重构使用的 normalized key                         |
| `g_ring`        |   `[HV,L]` | FP32             | log-decay                                             |
| `raw_v_ring`    | `[HV,L,V]` | activation dtype | exact fold 使用的原始 value input                     |
| `raw_k_ring`    |  `[H,L,K]` | activation dtype | L2 normalization 之前的 key                           |
| `beta_ring`     |   `[HV,L]` | FP32             | 重新计算 `d_i` 的 writing strength                    |
| cursor          |      `[3]` | int32            | committed length、circular base、next-step flush flag |

transformed rings 用于普通 verification/重构的快速路径，raw rings 是 checkpoint
写回的权威输入。`q` 不影响 `S_(i-1) -> S_i` 的状态转移，只影响 `y_i`，因此不用
缓存。MTP candidate 可以在 rejection sampling 前写入 ring，但只有 committed
cursor 覆盖的前缀可见。

虽然已经有 `raw_k_ring`，仍然保留 `k_ring` 是有意的。高频 non-flush 路径在每个
verify step 都直接使用已经归一化的 `(d_ring,k_ring,g_ring)` closed form；低频
exact-fold 路径只在物化 persistent checkpoint 时使用
`(raw_v_ring,raw_k_ring,g_ring,beta_ring)`。如果删除 `k_ring`，每次 verify 都要
重新归一化全部 cached raw key，或者增加等价的 normalized scratch buffer。因此这
可以作为显存/性能重新设计进行评估，但不是零成本删除。`d_ring` 与 `raw_v_ring`
同时存在也是同样的 hot/cold path 分工。

[Qwen3.5-35B-A3B][qwen35-model] TP1 中，`H=16`、`HV=32`、`K=V=128`，
共有 30 个 GDN layer。Standard `L=16` 时，每 layer/slot 的 shape 是
`S0[32,128,128]`、`d[32,16,128]`、`k[16,16,128]`、`g[32,16]`。
MTP3 `L=8` 时为 `S0[32,128,128]`、`d/raw_v[32,8,128]`、
`k/raw_k[16,8,128]`、`g/beta[32,8]`。已有 causal-convolution state 与上述
ring 分开：standard 为 `[8192,4]`，MTP3 为 `[8192,7]`。

### `S` 的重构与写回

对 Qwen3.5 预处理后的输入，定义：

```text
k_hat_i = L2Norm(k_raw_i)
q_hat_i = L2Norm(q_raw_i) / sqrt(K)
alpha_i = exp(g_i)
S_i^-   = alpha_i * S_(i-1)
d_i     = beta_i * (v_i - S_i^- * k_hat_i)
S_i     = S_i^- + d_i * k_hat_i^T
y_i     = S_i * q_hat_i
```

这就是 Gated DeltaNet recurrence；这里的 `d_i` 在 ReplaySSM 原理文章中也记作
`u_i`。

设 `S0` 是 checkpoint，standard ring 中已有 `m` 条 committed record，并定义
`G_j = sum_(r=1)^j g_r`。任意中间逻辑状态都可以在不读取 raw value 的情况下重构：

```text
S_j = exp(G_j) * S0
      + sum_(i=1)^j exp(G_j - G_i) * d_i * k_hat_i^T,
      1 <= j <= m.
```

standard kernel 通常只计算该表达式与当前 `q`、`k` 的 contraction，不把完整中间
矩阵写到 HBM。对当前 token `c`：

```text
d_c = beta_c * (v_c - alpha_c * S_m * k_hat_c)
y_c = alpha_c * (S_m * q_hat_c)
      + d_c * (k_hat_c^T * q_hat_c)
```

当 ring 写满或发生 forced flush 时，只物化并写回一次完整矩阵：

```text
S0 <- alpha_c * S_m + d_c * k_hat_c^T
```

随后 standard ring cursor 清零。因此完整 state 是每次 flush 写回一次，而不是每个
token 都写回。

MTP checkpoint 写回不会把低精度 transformed `d_ring` 当作权威数据，而是按
circular ring 的逻辑顺序读取 committed raw records，并在 FP32 中执行 closed-loop
fold：

```text
S <- S0
for i in committed_ring_order:
    k_hat_i = L2Norm(raw_k_i)
    S^-     = exp(g_i) * S
    d_i     = beta_i * (raw_v_i - S^- * k_hat_i)
    S       = S^- + d_i * k_hat_i^T
S0 <- S
```

中间 `S_i` 只存在于 kernel register，最终 `S0` 才写入 HBM。verify window 内也按
相同 recurrence 依次得到 candidate 的中间状态和输出，但相应 ring write 先保持
tentative。rejection sampling 完成后，device commit kernel 只按 accepted
target-input 数推进 cursor，因此 rejected suffix 不可见。如果需要 flush，也只在
写入新 speculative window 前 fold 此前已经 committed 的 raw prefix。

## 模式选择

LMDeploy 根据已有投机解码配置自动选择 ReplaySSM 协议，不另外提供 MTP
ReplaySSM 开关。

| 配置                                               | 行为              |
| -------------------------------------------------- | ----------------- |
| `enable_replayssm=False`                           | 现有基线路径      |
| 开启 Replay，未配置投机解码                        | 标准 ReplaySSM    |
| 开启 Replay，同时使用 `qwen3_5_mtp`                | MTP ReplaySSM     |
| GLM-5.3 Flash 开启 Replay，同时使用 `deepseek_mtp` | KDA MTP ReplaySSM |
| 开启 Replay，同时使用其他投机算法                  | 启动时报错        |

## 基本用法

### 命令行

标准解码接口：

```bash
lmdeploy serve api_server /path/to/qwen3.5-model \
  --backend pytorch \
  --enable-replayssm \
  --replayssm-buffer-len 16
```

Qwen3.5 MTP、三个 draft token 的接口：

```bash
lmdeploy serve api_server /path/to/qwen3.5-model \
  --backend pytorch \
  --enable-replayssm \
  --replayssm-buffer-len 8 \
  --speculative-algorithm qwen3_5_mtp \
  --speculative-num-draft-tokens 3
```

### Python API

标准解码：

```python
from lmdeploy import PytorchEngineConfig, pipeline

def main():
    backend_config = PytorchEngineConfig(
        enable_replayssm=True,
        replayssm_buffer_len=16,
    )
    pipe = pipeline('/path/to/qwen3.5-model', backend_config=backend_config)
    # 使用 pipe(...)，结束后关闭进程内 pipeline。
    pipe.close()


if __name__ == '__main__':
    main()
```

MTP 解码：

```python
from lmdeploy import PytorchEngineConfig, pipeline
from lmdeploy.messages import SpeculativeConfig

def main():
    backend_config = PytorchEngineConfig(
        enable_replayssm=True,
        replayssm_buffer_len=8,
    )
    speculative_config = SpeculativeConfig(
        method='qwen3_5_mtp',
        num_speculative_tokens=3,
    )
    pipe = pipeline(
        '/path/to/qwen3.5-model',
        backend_config=backend_config,
        speculative_config=speculative_config,
    )
    # 使用 pipe(...)，结束后关闭进程内 pipeline。
    pipe.close()


if __name__ == '__main__':
    main()
```

### 选择 buffer 长度

`replayssm_buffer_len` 只接受 `8`、`16` 或 `32`，默认值为 `16`。

- 较大的 ring 能降低 flush 频率，但每步需要读取更长历史，并占用更多状态缓存
  显存。
- 较小的 ring 减少 ring 显存和重放工作量，但会更频繁地物化完整 checkpoint。
- MTP 要求 `L >= 2 * (num_speculative_tokens + 1)`；因此 MTP3 至少需要
  `L=8`。
- 当前 H200 验证使用标准解码 `L=16`、MTP3 `L=8`。它们是经过验证的起点，
  不是对所有模型和负载都最优的固定值。

LMDeploy 初始化时会记录所选模式、每个 state slot 的对齐字节数、状态缓存总字节
和实际 cache geometry。确定部署并发数之前，应将这些值与 flag-off 配置比较。

### 回退

设置 `enable_replayssm=False`，或者移除 `--enable-replayssm`，即可回到现有
循环状态实现。flag-off 路径始终是正确性和性能对照组。

## 参考资料

- [ReplaySSM：Cache SSM Inputs, Not State][replayssm-blog]
- [ReplaySSM 参考实现][replayssm-code]
- [Gated Delta Networks: Improving Mamba2 with Delta Rule][gdn-paper]
- [Qwen3.5-35B-A3B 模型配置][qwen35-model]

[gdn-paper]: https://arxiv.org/abs/2412.06464
[qwen35-model]: https://huggingface.co/Qwen/Qwen3.5-35B-A3B
[replayssm-blog]: https://dao-lab.ai/blog/2026/replayssm/
[replayssm-code]: https://github.com/Johnny-Liou/ReplaySSM
