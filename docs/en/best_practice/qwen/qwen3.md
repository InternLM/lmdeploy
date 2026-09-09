# Qwen3

## 1. Model Introduction

Qwen3 is the latest generation of large language models in the Qwen series developed by Alibaba, offering significant improvements in instruction following, reasoning, multilingual understanding, and tool usage.

The Qwen3 series provides models in both **Dense** and **MoE** (Mixture-of-Experts) architectures:

|      Model      | Type  | Parameters | Active Parameters |
| :-------------: | :---: | :--------: | :---------------: |
|   Qwen3-0.6B    | Dense |    0.6B    |       0.6B        |
|   Qwen3-1.7B    | Dense |    1.7B    |       1.7B        |
|    Qwen3-4B     | Dense |     4B     |        4B         |
|    Qwen3-8B     | Dense |     8B     |        8B         |
|    Qwen3-14B    | Dense |    14B     |        14B        |
|    Qwen3-32B    | Dense |    32B     |        32B        |
|  Qwen3-30B-A3B  |  MoE  |    30B     |        3B         |
| Qwen3-235B-A22B |  MoE  |    235B    |        22B        |

Key features:

- **Extended context length**: Up to 256K tokens for long-context understanding and reasoning.
- **Flexible deployment**: Available in Base, Instruct, and Thinking editions.
- **Tool calling**: Built-in support for function calling and agent workflows.
- **Multilingual**: Broad multilingual knowledge coverage.

For more details, please refer to the [Qwen3 GitHub Repository](https://github.com/QwenLM/Qwen3).

## 2. Model Deployment

### 2.1 Basic Configuration

The Qwen3 series is fully supported by LMDeploy with both TurboMind and PyTorch backends. Recommended launch configurations vary by hardware and model size.

**Interactive Command Generator**: Use the configuration selector below to automatically generate the appropriate deployment command for your hardware platform, model size, quantization method, and capabilities.

```{raw} html
<div id="lmdeploy-config-generator" data-model-config="qwen3"></div>
```

### 2.2 Configuration Tips

- **Backend Selection**: TurboMind is the default high-performance backend. Use PyTorch backend (`--backend pytorch`) for broader model format compatibility and features like LoRA adapters.
- **Tensor Parallelism (`--tp`)**: Set based on model size and available GPUs. Larger models (32B+) typically require multi-GPU setups.
- **KV Cache Memory (`--cache-max-entry-count`)**: Controls the percentage of free GPU memory used for KV cache (default: 0.8). Lower this value if you encounter OOM errors.
- **Session Length (`--session-len`)**: Defaults to model's max length. Set explicitly to conserve memory, e.g., `--session-len 32768`.
- **Prefix Caching (`--enable-prefix-caching`)**: Enables automatic prefix caching for improved throughput when serving repeated prompt patterns.
- **Quantization**: LMDeploy supports AWQ 4-bit (`--model-format awq`) and KV cache quantization (`--quant-policy 4` or `--quant-policy 8`) for Qwen3 models.

## 3. Model Invocation

### 3.1 Basic Usage

For basic API usage, please refer to:

- [OpenAI Compatible Server](../../llm/api_server.md)
- [Pipeline (Offline Inference)](../../llm/pipeline.md)

### 3.2 Reasoning Parser

Qwen3 Thinking models support reasoning mode. Enable the reasoning parser during deployment to separate the thinking and content sections:

```shell
lmdeploy serve api_server Qwen/Qwen3-32B-Thinking --reasoning-parser qwen-qwq
```

For detailed usage and examples, see [Reasoning Outputs](../../llm/api_server_reasoning.md).

### 3.3 Tool Calling

Qwen3 supports tool calling capabilities. Enable the tool call parser:

```shell
lmdeploy serve api_server Qwen/Qwen3-32B-Instruct --tool-call-parser qwen3
```

For detailed usage and examples, see [Tools](../../llm/api_server_tools.md).
