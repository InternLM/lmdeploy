# GLM-4

## 1. Model Introduction

GLM-4 is a series of large language models developed by Tsinghua University (THUDM). The series spans from compact 9B models to large-scale 754B models, offering strong multilingual and reasoning capabilities.

|     Model     | Parameters | Architecture |
| :-----------: | :--------: | :----------: |
|     GLM-4     |     9B     |    Dense     |
|  GLM-4-0414   |     9B     |    Dense     |
|    GLM-4.5    |    355B    |     MoE      |
|  GLM-4.5-Air  |    106B    |     MoE      |
| GLM-4.7-Flash |    30B     |    Dense     |
|     GLM-5     |    754B    |     MoE      |

Key features:

- **Scalable Architecture**: From 9B dense to 754B MoE models.
- **Strong Multilingual Support**: Excellent Chinese and English capabilities.
- **Tool Calling**: Built-in function calling support.
- **Vision-Language Models**: GLM-4V variants available for multimodal tasks.

For more details, please refer to the [GLM GitHub Repository](https://github.com/THUDM).

## 2. Model Deployment

### 2.1 Basic Configuration

GLM-4 models are supported by LMDeploy with both TurboMind (9B models) and PyTorch backends. Use the interactive generator below to create your deployment command.

**Interactive Command Generator**:

```{raw} html
<div id="lmdeploy-config-generator" data-model-config="glm4"></div>
```

### 2.2 Configuration Tips

- **Backend Selection**: GLM-4 (9B) works with both TurboMind and PyTorch backends. Larger models (GLM-4.5, GLM-5) require the PyTorch backend.
- **Tensor Parallelism (`--tp`)**: GLM-4 (9B) can run on a single 80G GPU. GLM-4.5 (355B) requires multi-GPU setups.
- **Quantization**: AWQ quantization is supported for GLM-4 (9B) models on TurboMind backend.
- **Session Length (`--session-len`)**: Set explicitly to conserve memory, e.g., `--session-len 32768`.

## 3. Model Invocation

### 3.1 Basic Usage

For basic API usage, please refer to:

- [OpenAI Compatible Server](../../llm/api_server.md)
- [Pipeline (Offline Inference)](../../llm/pipeline.md)
