# DeepSeek

## 1. Model Introduction

DeepSeek is a series of powerful open-source large language models developed by DeepSeek AI. The series features Mixture-of-Experts (MoE) architecture for efficient inference with massive parameter counts.

|      Model       | Parameters | Architecture |
| :--------------: | :--------: | :----------: |
| DeepSeek-V2-Lite |    16B     |     MoE      |
|   DeepSeek-V2    |    236B    |     MoE      |
|  DeepSeek-V2.5   |    236B    |     MoE      |
|   DeepSeek-V3    |    685B    |     MoE      |
|  DeepSeek-V3.2   |    685B    |     MoE      |

Key features:

- **MoE Architecture**: Efficient inference through sparse activation of expert modules.
- **Large-scale Models**: Up to 685B total parameters with efficient activated parameter counts.
- **Strong Reasoning**: Deep reasoning capabilities, especially with DeepSeek-R1 reasoning mode.

For more details, please refer to the [DeepSeek GitHub Repository](https://github.com/deepseek-ai).

## 2. Model Deployment

### 2.1 Basic Configuration

DeepSeek models are supported by LMDeploy with the PyTorch backend. Use the interactive generator below to create your deployment command.

**Interactive Command Generator**:

```{raw} html
<div id="lmdeploy-config-generator" data-model-config="deepseek"></div>
```

### 2.2 Configuration Tips

- **Backend**: DeepSeek models use the PyTorch backend (`--backend pytorch`).
- **Tensor Parallelism (`--tp`)**: DeepSeek-V3 (685B) requires at least 8×80G GPUs. Smaller models like V2-Lite (16B) can run on a single GPU.
- **Session Length (`--session-len`)**: Set explicitly to conserve memory, e.g., `--session-len 32768`.
- **Cache Management (`--cache-max-entry-count`)**: Lower this value if you encounter OOM errors.

## 3. Model Invocation

### 3.1 Basic Usage

For basic API usage, please refer to:

- [OpenAI Compatible Server](../../llm/api_server.md)
- [Pipeline (Offline Inference)](../../llm/pipeline.md)

### 3.2 Reasoning Parser

DeepSeek models support reasoning mode via the DeepSeek-R1 reasoning parser:

```shell
lmdeploy serve api_server deepseek-ai/DeepSeek-V3 --backend pytorch --reasoning-parser deepseek-r1
```

For detailed usage and examples, see [Reasoning Outputs](../../llm/api_server_reasoning.md).
