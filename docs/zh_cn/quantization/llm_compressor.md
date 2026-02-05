# llm-compressor 支持

本指南旨在介绍如何使用 LMDeploy 的 TurboMind 推理引擎，运行经由 [vllm-project/llm-compressor](https://github.com/vllm-project/llm-compressor)工具量化后的模型。
目前支持的 `llm-compressor` 量化模型包括：

- int4 量化（例如 AWQ、GPTQ）

上述量化模型通过 TurboMind 引擎可以在以下 NVIDIA GPU 架构上运行：

| Compute Capability | Micro-architecture | GPUs                            |
| ------------------ | ------------------ | ------------------------------- |
| 7.0                | Volta              | V100                            |
| 7.2                | Volta              | Jetson Xavier                   |
| 7.5                | Turing             | GeForce RTX 20 series, T4       |
| 8.0                | Ampere             | A100, A800, A30                 |
| 8.6                | Ampere             | GeForce RTX 30 series, A40, A10 |
| 8.7                | Ampere             | Jetson Orin                     |
| 8.9                | Ada Lovelace       | GeForce RTX 40 series, L40, L20 |
| 9.0                | Hopper             | H20, H200, H100, GH200          |
| 12.0               | Blackwell          | GeForce RTX 50 series           |

LMDeploy 将持续跟进并扩展对 `llm-compressor` 项目的支持。

本文的其余部分由以下章节组成：

<!-- toc -->

- [模型量化](#模型量化)
- [模型部署](#模型部署)
- [精度评测](#精度评测)

<!-- tocstop -->

## 模型量化

`llm-compressor` 提供了丰富的模型量化[用例](https://github.com/vllm-project/llm-compressor/tree/main/examples)，请参考其教程选择 LMDeploy 支持的量化算法，完成模型量化工作。
LMDeploy 也内置了通过 `llm-compressor` 对 Qwen3-30B-A3B 进行 AWQ 量化的[脚本](https://github.com/InternLM/lmdeploy/blob/main/examples/lite/qwen3_30b_a3b_awq.py)，供大家进行参考：

```shell
# 创建 conda 环境
conda create -n lmdeploy python=3.10 -y
conda activate lmdeploy

# 安装 llm-compressor
pip install llm-compressor

# 下载 lmdeploy 源码，运行量化用用例
git clone https://github.com/InternLM/lmdeploy
cd lmdeploy
python examples/lite/qwen3_30b_a3b_awq.py --work-dir ./qwen3_30b_a3b_awq

```

在接下来的章节中，我们以此量化模型为例，介绍模型部署、评测精度等方法

## 模型部署

### 离线推理

量化后的模型，通过以下几行简单的代码，可以实现离线批处理：

```python
from lmdeploy import pipeline, TurbomindEngineConfig
engine_config = TurbomindEngineConfig()
with pipeline("./qwen3_30b_a3b_4bit", backend_config=engine_config) as pipe:
    response = pipe(["Hi, pls intro yourself", "Shanghai is"])
    print(response)
```

关于 pipeline 的详细介绍，请参考[这里](https://lmdeploy.readthedocs.io/zh-cn/latest/llm/pipeline.html)

### 在线服务

LMDeploy api_server 支持把模型一键封装为服务，对外提供的 RESTful API 兼容 openai 的接口。以下为服务启动的示例：

```shell
lmdeploy serve api_server ./qwen3_30b_a3b_4bit --backend turbomind
```

服务默认端口是23333。在 server 启动后，你可以通过 openai SDK 访问服务。关于服务的命令参数，以及访问服务的方式，可以阅读[这份](https://lmdeploy.readthedocs.io/zh-cn/latest/llm/api_server.html)文档

## 精度评测

我们将 Qwen3-8B (Dense) 与 Qwen3-30B-A3B (MoE) 的 AWQ 对称/非对称量化模型通过 LMDeploy 部署为服务，并使用 [opencompass](https://github.com/open-compass/opencompass) 在多个学术数据集上评测。结果显示：Qwen3-8B 的非对称量化整体优于对称量化，而 Qwen3-30B-A3B 在两种量化方式间差异不显著；Qwen3-8B 在对称/非对称量化下与 BF16 模型的精度差异小于 Qwen3-30B-A3B。与 BF16 相比，量化模型在长输出数据集，比如 aime2025 (平均 17,635 tokens)、LCB (平均 14,157 tokens)，精度下降更明显；在中短输出数据集，比如 ifeval (平均 1,885 tokens)，mmlu_pro (平均 2,826)，精度符合预期。

| dataset           | Qwen3-8B |         |          | Qwen3-30B-A3B |         |          |
| ----------------- | -------- | ------- | -------- | ------------- | ------- | -------- |
|                   | bf16     | awq sym | awq asym | bf16          | awq sym | awq asym |
| ifeval            | 85.58    | 83.73   | 85.77    | 86.32         | 84.10   | 84.29    |
| hle               | 5.05     | 5.05    | 5.24     | 7.00          | 5.47    | 5.65     |
| gpqa              | 59.97    | 56.57   | 59.47    | 61.74         | 57.95   | 57.07    |
| aime2025          | 69.48    | 64.38   | 63.96    | 73.44         | 64.79   | 66.67    |
| mmlu_pro          | 73.69    | 71.73   | 72.34    | 77.85         | 75.77   | 75.69    |
| LCBCodeGeneration | 50.86    | 44.10   | 46.95    | 56.67         | 50.86   | 49.24    |

复现方式可以参考[这份](https://lmdeploy.readthedocs.io/zh-cn/latest/benchmark/evaluate_with_opencompass.html)文档
