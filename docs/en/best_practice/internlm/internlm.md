# InternLM

## 1. Model Introduction

InternLM is a series of large language models developed by Shanghai AI Laboratory and SenseTime. The series spans multiple generations with progressive improvements in reasoning, code generation, and tool usage.

|     Model      | Parameters | Architecture |
| :------------: | :--------: | :----------: |
|  InternLM2-7B  |     7B     |    Dense     |
| InternLM2-20B  |    20B     |    Dense     |
| InternLM2.5-7B |     7B     |    Dense     |
|  InternLM3-8B  |     8B     |    Dense     |

Key features:

- **Strong Reasoning**: Excellent performance on reasoning and math benchmarks.
- **Tool Calling**: Built-in function calling and agent capabilities.
- **Code Generation**: Strong code generation and understanding abilities.
- **Long Context**: Support for extended context windows.

For more details, please refer to the [InternLM GitHub Repository](https://github.com/InternLM/InternLM).

## 2. Model Deployment

### 2.1 Basic Configuration

InternLM models are fully supported by LMDeploy with both TurboMind and PyTorch backends. Use the interactive generator below to create your deployment command.

**Interactive Command Generator**:

```{raw} html
<div id="lmdeploy-config-generator" data-model-config="internlm"></div>
```

### 2.2 Configuration Tips

- **Backend Selection**: TurboMind is the default high-performance backend. Use PyTorch backend (`--backend pytorch`) for broader compatibility.
- **Tensor Parallelism (`--tp`)**: InternLM2-20B may require 2 GPUs for BF16 inference. Smaller models (7B/8B) fit on a single GPU.
- **Quantization**: AWQ quantization (`--model-format awq`) and KV cache INT8 (`--quant-policy 8`) are supported.
- **Session Length (`--session-len`)**: Set explicitly to conserve memory, e.g., `--session-len 32768`.

## 3. Model Invocation

### 3.1 Basic Usage

For basic API usage, please refer to:

- [OpenAI Compatible Server](../../llm/api_server.md)
- [Pipeline (Offline Inference)](../../llm/pipeline.md)

### 3.2 Reasoning Parser

InternLM models support reasoning mode via the `intern-s1` reasoning parser:

```shell
lmdeploy serve api_server internlm/internlm3-8b-instruct --reasoning-parser intern-s1
```

For detailed usage and examples, see [Reasoning Outputs](../../llm/api_server_reasoning.md).

### 3.3 Tool Calling

InternLM supports tool calling capabilities. Enable the tool call parser:

```shell
lmdeploy serve api_server internlm/internlm3-8b-instruct --tool-call-parser internlm
```

For detailed usage and examples, see [Tools](../../llm/api_server_tools.md).
