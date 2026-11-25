# 序列并行

在单卡显存不足以部署模型的时候，通常会以 `TP` 的方式进行部署，而这一般要求 `num_key_value_heads` 被 `TP` 整除。如果要以 `TP > num_key_value_heads` 的方式进行部署，需要创建 kv-heads 的副本，以满足整除需求。但是这样会有两个缺点：

1. 可用的 kvcache 数量减半，进而减少请求最大推理长度
2. 降低推理的最大 batch 数量，减少吞吐量。

为了解决这个问题，TurboMind 推理后端支持设置 `attn_dp_size`，避免了创建 kv-heads 的副本，但是这会引入数据的不均衡性。为了消除数据的不均衡，TurboMind 支持了序列并行，支持将 kv_cache 交错存储到不同的 cp_rank 上，例如

```
cp_rank=2, prompt_len=5, generation_len=4
kv_cache stored on cp_rank0: 0, 2, 4, 6, 8
kv_cache stored on cp_rank1: 1, 3, 5, 7
```

在序列并行模式下，`cache_block_seq_len` 仍表示每个 rank 的一个 k/v cache block 实际存放的 token 数。调度器中对应的逻辑块覆盖 `cache_block_seq_len * cp` 个全局 token。因此每个 rank 上 k/v block 的显存大小不变，而整块前缀复用和只读 cache 边界按更大的全局跨度计算。

## 使用说明

以 `Intern-S1` / `Qwen3-235B-A22B` 为例，他们的 `num_key_value_heads` 为 4，若要用 `TP=8` 的方式部署，并避免 kv_cache 的拷贝，可以用如下的方式部署

```
lmdeploy serve api_server internlm/Intern-S1 --tp 8 --cp 2

lmdeploy serve api_server Qwen/Qwen3-235B-A22B --tp 8 --cp 2
```

## PyTorch 解码上下文并行（DCP）

PyTorch 使用 `--dcp`，与 TurboMind 的 `--cp` 区分。DCP 在现有 attention TP 组内分片原本重复的 KV cache，不增加 GPU 数量。支持 FlashMLA 的 dense MLA、sparse DSA，以及具有 KV head 副本的 GQA。

```bash
lmdeploy serve api_server <mla-model> --backend pytorch --tp 4 --dcp 2

# 这两个 Qwen3 模型有 4 个 KV heads；TP8 时每个 KV head 有两个副本。
lmdeploy serve api_server Qwen/Qwen3-30B-A3B --backend pytorch --tp 8 --dcp 2
lmdeploy serve api_server Qwen/Qwen3-235B-A22B --backend pytorch --tp 8 --dcp 2
```

Python 接口可设置 `PytorchEngineConfig(tp=4, dcp=2)`；默认 `dcp=1` 表示关闭 DCP。

### 支持范围与限制

- MLA 要求 NVIDIA Hopper/SM90、FlashMLA 和 BF16 激活。Sparse DSA 还需要兼容的 DeepGEMM、TileLang top-k 算子（top-k 为 512 或 2048）。
- GQA decode 使用 Triton；prefill 在可用时使用 FlashAttention-3，否则使用 Triton。支持 BF16/FP16 激活、未量化或 per-tensor FP8 KV cache（`--quant-policy fp8` / `fp8_e5m2`）。不支持 INT8、INT4、TurboQuant、ALiBi 和 attention sinks。
- `dcp` 必须整除 attention TP 大小及 KV head 副本数。同一个 DCP 组必须持有相同 KV head。目前要求 `dp=1`、`ep=1`。
- 兼容的 MLA 模型支持 prefix caching 和 `deepseek_mtp`，包括 DeepSeek V3/V3.1、DeepSeek V3.2 和 GLM DSA。Sparse MLA 也支持 FP8 KV cache。
- GQA prefill 保持本地 Q heads，按有界分块收集 cached K/V；decode 收集 Q，按 softmax 归一化因子合并各上下文分片的输出。
- 不支持滑窗、MemDecode、PD 分离、外部 KV-cache connector 和 TileLang attention backend。GQA 的滑窗检查逐层执行，覆盖 Gemma 3 这类局部/全局 attention 混合模型。非因果 GQA 和 SDAR/dLLM block-sparse attention（`dllm_block_length > 1`）也会被拒绝。
- DFlash 的 draft cache 写入和非因果 block attention 尚未实现 DCP，因此禁止 `dcp>1`。Llama EAGLE3 的 draft 未分片却共用 DCP target 页表时也会被拒绝。这些组合应使用 `dcp=1`；共用兼容 cache 布局的 speculative 路径（如 `deepseek_mtp`）仍可使用。

### 通信策略与兼容性

`--communication-backend nccl` 为默认值；未显式设置已有优化环境变量时使用原生 collective。`auto` 启用符合条件的 symmetric-memory gather 和 TP all-reduce，不满足输入或拓扑条件时回退到原生通信。

原有 `LMDEPLOY_ENABLE_FLASHINFER_ALLREDUCE`、`LMDEPLOY_ENABLE_SYMM_MEM_ALLREDUCE`、`LMDEPLOY_ENABLE_SYMM_MEM_LMHEAD` 保留各自的 TP/LM-head 作用范围。显式开启 FlashInfer 时，它优先于 `auto` 的 TP all-reduce 选择，并保留 DeepSeek V3.2 的 all-reduce + residual + RMSNorm 分派。两个原有 all-reduce 开关不能同时开启；这些 TP 开关不会开启 DCP gather 优化。

`LMDEPLOY_SYMM_MEM_LMHEAD_MAX_MB` 继续限制每个 rank 的 LM-head gather workspace，包括 `auto` 模式；默认 64 MiB、最小 1 MiB，不影响 DCP query/prefix workspace。组内各 rank 的通信配置必须一致。

### GQA FP8 prefill 精度语义

DCP prefill 使用当前块未量化的 K/V，并与从 FP8 cache 反量化的 prefix K/V 合并；decode 的全部 K/V 来自 cache。这延续 MLA DCP 的策略，但与普通 GQA prefill 从 cache 读取全部 K/V 的方式不同。因此开启 DCP、改变分块边界或复用 prefix 都可能改变哪些 token 经历 FP8 舍入，不保证逐位一致；这种差异本身不能归因于通信错误。

`tests/pytorch/test_dcp_gqa_reference.py` 覆盖 BF16、E4M3、E5M2 下的完整/分块 prefill、prefix cache 页复用、单/多 token decode。测试独立检查 cache 分片，并分别比较使用相同 K/V 精度的 FP32 attention reference 和非 DCP GQA 路径。对于测试中固定的有界输入，reference 的容差为 `rtol=atol=0.02`；非 DCP 和分块边界比较的容差分别为 0.02（BF16）、0.04（E4M3）、0.08（E5M2）。这些是该测试样例的回归阈值，不是模型精度保证。
