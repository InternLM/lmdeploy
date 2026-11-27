# 前缀缓存（PyTorch）

本文介绍 PyTorch 后端的本地前缀缓存，以及与 Mooncake Store 共用的 hybrid
state checkpoint 保存策略。通过 `PytorchEngineConfig.enable_prefix_caching=True`
启用本地缓存；Mooncake Store 需通过 `kv_transfer_config` 单独配置。
TurboMind 后端请参阅其[前缀缓存配置](../inference/turbomind_config.md#前缀缓存开关)。

## Hybrid attention 模型的 prefill checkpoint

Hybrid attention 模型复用前缀时，需要同时具备 full-attention KV blocks
和该前缀边界上的 linear-attention state。

- `prefill_save_alignment`（CLI 参数 `--prefill-save-alignment`，默认 1024）统一控制
  本地 prefix caching 和 Mooncake Store 的 prefill 状态快照分块边界，取值必须是缓存块大小的正整数倍。设置为 512、
  token 预算为 8192 时，首次处理 800-token 纯文本输入会切成 512 和 288 两段，
  同时开启两种缓存也一样。仅含 full attention 的模型忽略此参数；对齐不代表每到一个倍数都保存快照。
- `prefix_cache_state_budget` 为 checkpoint 预留额外状态槽，会增加显存占用。
  设为零表示不额外预留，仍可复用空闲的运行时状态槽。

### 仅保存最后一个 checkpoint

`prefill_save_last_only=True`（CLI 参数 `--prefill-save-last-only`，默认关闭）
每轮输入仅保存最后一个对齐 `prefill_save_alignment` 的可复用边界。
该边界至少保留一个输入 token，遇到多模态 span 时向前回退，并考虑 MTP 的输入偏移。
此前的 chunk 按正常 token 预算推进；到达目标边界的 chunk 同时保存本地 state checkpoint，
并向 Mooncake Store 发送截至该边界尚未保存的全部 full-attention KV blocks 和对应 state snapshot。
两种缓存仍需分别启用。

例如显式将对齐设为 8192，输入长度 20,000 时保存在 16,384；
若图像占据 \[16,000, 18,000)，则回退到 8192。输入长度恰为 32,768 时保存在 24,576，
留出后缀用于 prefix 复用。没有正数安全边界或缓存命中已到达该边界时，本轮不新增保存。
旧轮次的 checkpoint 仍按原有规则淘汰，snapshot 槽不足时仍按现有策略跳过。

该选项可减少追加式 agent 多轮对话的 checkpoint 拷贝和 Store 提交次数，
代价是编辑历史或分叉对话时可复用的中间前缀更少。仅含 full attention 的模型忽略此参数。

支持的 MTP 方法在本地和 Store 匹配时均使用 token lookahead，不采用整块回退，
因此远端复用无需 state 边界之后的额外 KV blocks。

## GLM-5.3-Flash

GLM-5.3-Flash 的普通自回归解码和 [`deepseek_mtp`](./spec_decoding.md#deepseek-mtp) 都可以启用
`PytorchEngineConfig.enable_prefix_caching`。下面是 TP4 的纯文本示例，
请根据硬件选择 TP 和缓存预算。将 `speculative_config` 设为 `None` 即可关闭
MTP，同时保留 prefix caching。

```python
from lmdeploy import GenerationConfig, PytorchEngineConfig, pipeline
from lmdeploy.messages import SpeculativeConfig

if __name__ == '__main__':
    engine = PytorchEngineConfig(
        tp=4,
        max_batch_size=4,
        enable_prefix_caching=True,
        prefix_cache_state_budget=8,
        max_prefill_token_num=512,
        language_model_only=True,
    )
    spec = SpeculativeConfig(method='deepseek_mtp', num_speculative_tokens=3)
    with pipeline('/path/to/GLM-5.3-Flash', backend_config=engine,
                  speculative_config=spec) as pipe:
        notes = 'A rectangle has four sides and four right angles. ' * 80
        prompt = notes + '\nHow many sides does a rectangle have?'
        generation = GenerationConfig(max_new_tokens=128, do_sample=False)
        cold = pipe(prompt, gen_config=generation)
        warm = pipe(prompt, gen_config=generation)
        print(cold.cached_tokens, warm.cached_tokens)
```

### 配置限制

- 缓存块大小为 64 tokens。手动设置 `num_gpu_blocks` 时按此计数，而不是按 KPool 条目计数。
- `prefix_cache_decode_state_interval` 取正值时必须是 64 的倍数。
  使用 `deepseek_mtp` 保存 decode checkpoint 时，draft 模型必须只有一层，且不使用状态缓存。
- 暂不支持 PD 迁移。

原生 FP8 CUDA 路径也支持 DP/EP 服务，例如 `--tp 2 --dp 2 --ep 4`。
使用 `--piecewise-cudagraph-max-tokens 512 --max-prefill-token-num 512`
启用文本 prefill PCG。启动 warmup 负责准备 plan；禁用 warmup 时未准备的
prefill 走 eager，包含视觉输入的 chunk 也回退到 eager。MTP draft prefill
仍走 eager，受支持的 decode 使用普通 CUDA graph。这些选项与 prefix caching 独立。

用 `Response.cached_tokens` 检查实际命中，本地和 Store 的累计命中率见[缓存命中指标](./metrics.md)。
Prefill 分块、PCG padding、batch 形状或并行拓扑变化可能改变 FP8 的生成轨迹；
应同时检查任务 accuracy、MTP accept rate 和缓存命中，不能默认逐 token 完全相同或一定获得性能收益。

## 多实例共享 Mooncake Store

启动一个 [Mooncake master](https://kvcache-ai.github.io/Mooncake/deployment/mooncake-store-deployment-guide.html)。
在每个节点将 `MOONCAKE_CONFIG_PATH` 指向各 worker 可读的 JSON 文件。以 RDMA 为例：

```json
{
  "master_server_address": "store-master:50051",
  "metadata_server": "P2PHANDSHAKE",
  "protocol": "rdma",
  "global_segment_size": "4GB",
  "local_buffer_size": "4GB"
}
```

各节点填写同一个可达的 master 地址（不要用 `127.0.0.1`）。启用 Mooncake 的 RDMA 自动发现时可省略 `device_name`；手动选卡时填写本机的 RDMA 设备名。
两项容量均按**每个 Store worker** 计算。按示例默认值，单节点 8 个 worker 共贡献 32 GiB 共享缓存，另需为本地传输缓冲区预留 32 GiB。
权重和缓存布局相同的副本，在启动 PyTorch 服务时使用相同的 connector 参数：

```shell
--kv-transfer-config '{"kv_connector":"MooncakeStoreConnector","kv_role":"kv_both","kv_connector_extra_config":{"model_name":"my-model","cache_prefix":"run-42"}}'
```

### RL 权重更新

训练与推理共卡时，在 `kv_connector_extra_config` 中按需添加：

- 设置 `rl_weight_versioning=True`，并为初始权重指定非空的 `weight_version`。不同权重不能共用版本号。
- 跨副本复用缓存时，设置 `rl_share_across_engines=True`，并让各副本使用相同的权重版本。引擎休眠后，须先提交权重更新，再唤醒 KV 缓存。
- 若模型支持返回路由专家 ID，可设置 `store_routed_experts=True`，将这些 ID 与 KV 一起缓存。这会额外占用输出 rank 的显存。

按 [Ray 权重更新流程](./update_weights.md)更新权重时，保持 `cache_prefix` 不变。
每轮只在最后一个 bucket 的 `/update_weights` 请求（`finished=True`）中传入新权重的
`weight_version`（如 `step-1`）。共享缓存的所有 rank 和副本使用同一版本号。
如果同一引擎的所有 rank 都省略版本号，引擎会生成新的私有版本，不会沿用上一轮版本。
版本号校验失败不会回滚已加载的权重。
