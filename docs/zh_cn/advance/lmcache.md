# 在 TurboMind 中使用 LMCache

TurboMind 可以通过 [LMCache](https://github.com/LMCache/LMCache) 将 KV cache
存储在 GPU 显存之外，并在后续请求中取回复用。该集成通过 LMCache 多进程服务执行
LOOKUP、STORE 和 RETRIEVE。

本地前缀缓存与 LMCache 可以同时开启。本地前缀缓存复用 GPU 中的 KV，LMCache
可以补充本地不存在的缓存。复用外部 KV 能减少重复的 prefill 计算，但查询和传输也有开销。
应根据实际负载评估性能，尤其是在本地 GPU 缓存已经足够的情况下。

## 环境要求与安装

需要 Linux、NVIDIA GPU 和支持 LMCache 的 LMDeploy，后者的安装方法见
[安装文档](../get_started/installation.md)。LMCache 与 TurboMind 应运行在同一主机，
并使用相同的物理 GPU。

在已安装 PyTorch 和兼容 CUDA Toolkit 的 Python 环境中，安装基于 LMCache 0.5.5 的
TurboMind 适配版本：

```bash
git clone --branch tm https://github.com/irexyc/lmcache.git
cd lmcache
git checkout a41e1d0a5b8624a0bc7aa2fff468984845a4ca10
python -m pip install -r requirements/build.txt
LMCACHE_CUDA_MAJOR=12 python -m pip install -e . --no-build-isolation
```

以上使用 CUDA 12；CUDA 13 环境将 `LMCACHE_CUDA_MAJOR` 改为 `13`。

## 启动服务

### 1. 启动 LMCache

在第一个终端执行：

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3
lmcache server \
  --host 127.0.0.1 \
  --port 5558 \
  --chunk-size 256 \
  --l1-size-gb 512 \
  --eviction-policy LRU
```

等待服务端输出启动成功的信息。`--l1-size-gb 512` 配置了 512 GiB 的主机内存缓存，
主机可用内存不足时应调小该值。所选 GPU 应可供本次部署使用。

### 2. 启动 TurboMind API server

在第二个终端中使用相同的 GPU 配置：

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3
export MODEL_PATH=Qwen/Qwen3-30B-A3B
lmdeploy serve api_server "$MODEL_PATH" \
  --backend turbomind \
  --tp 4 \
  --enable-prefix-caching \
  --session-len 65536 \
  --lmcache-addr tcp://127.0.0.1:5558
```

`MODEL_PATH` 也可以替换为本地模型目录。API server 默认监听 23333 端口。
示例的 65,536-token 会话长度为 32K 首轮输入和后续多轮对话留出了空间，应按模型和负载选择。
`--async` 默认值为 1，无需额外设置。

通过常规的 [OpenAI 兼容接口](../llm/api_server.md#restful-api)发送请求即可，调用方
不需要增加 LMCache 参数。若要关闭 LMCache，在启动 API server 时省略 `--lmcache-addr`；
本地前缀缓存由独立开关控制。

## 使用 Python pipeline

保持 LMCache 服务运行，也可以使用 pipeline 代替 API server。在启动 Python 前设置
`CUDA_VISIBLE_DEVICES=0,1,2,3`。

```python
from lmdeploy import GenerationConfig, TurbomindEngineConfig, pipeline

backend_config = TurbomindEngineConfig(
    tp=4,
    enable_prefix_caching=True,
    session_len=65536,
    lmcache_addr='tcp://127.0.0.1:5558',
)

with pipeline('Qwen/Qwen3-30B-A3B', backend_config=backend_config) as pipe:
    response = pipe(
        'Explain how KV caching accelerates language model inference.',
        gen_config=GenerationConfig(max_new_tokens=64),
    )
    print(response.text)
```

省略 `lmcache_addr` 或设置为 `None` 即可关闭外部缓存。其他推理选项可参考
[pipeline 使用指南](../llm/pipeline.md)。

## 配置与限制

| 配置                                                | 作用                                                                    |
| --------------------------------------------------- | ----------------------------------------------------------------------- |
| `--lmcache-addr` / `lmcache_addr`                   | LMCache 的 ZeroMQ 地址，默认不设置，即关闭 LMCache。                    |
| `--enable-prefix-caching` / `enable_prefix_caching` | 本地 GPU 前缀缓存开关，与 LMCache 独立。                                |
| LMCache `--chunk-size`                              | 外部缓存传输的 token 粒度。TurboMind 默认块大小为 64 时，可设置为 256。 |
| LMCache `--l1-size-gb`                              | 主机内存缓存容量，单位 GiB，与 TurboMind 的 GPU 缓存容量独立。          |
| LMCache `--eviction-policy`                         | L1 缓存淘汰策略，示例使用 LRU。                                         |

LMCache chunk size 必须是 TurboMind 逻辑块大小的整数倍，后者为
`cache_block_seq_len * cp`，本文使用默认 `cp=1`。对于 GDN 模型，TurboMind 还会在
chunk 边界保存循环状态，不需要额外的 GDN 开关。GPU 缓存容量和前缀缓存配置详见
[TurboMind 配置](../inference/turbomind_config.md)。

请求 perplexity 或全部 token 的 logits/hidden states 时，需要执行提示词计算，因此会绕过
外部缓存；本地前缀缓存对这些输出模式也有独立限制。使用随请求变化的 RoPE base 的
Dynamic NTK 请求同样绕过外部缓存。本文介绍纯文本输入，多模态 fingerprint 冲突处理尚未完善。

更换模型、权重、TP 配置或缓存布局后，应启动新的 LMCache 实例。当前集成不提供针对
不同部署历史缓存的兼容性检查。
