# `llm-compressor` 支持

本指南旨在介绍如何使用 LMDeploy 的 TurboMind 推理引擎，运行经由 [vllm-project/llm-compressor](https://github.com/vllm-project/llm-compressor)工具量化后的模型。
目前支持的 `llm-compressor` 量化模型包括：

- 4-bit 非对称量化（例如 AWQ、GPTQ）

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
LMDeploy 也内置了通过 `llm-compressor` 对 Qwen3-30B-A3B 进行 AWQ 量化的[脚本](https://github.com/InternLM/lmdeploy/examples/lite/qwen3_30b_a3b_awq.py)，供大家进行参考：

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

我们把上述量化模型通过 LMDeploy 部署为服务后，使用 [opencompass](https://github.com/open-compass/opencompass) 评测了其在若干学术数据集上的精度，与原始模型相比，精度误差在可接受范围之内：

| dataset           | Qwen3-30B-A3B | Qwen3-30B-A3B awq | diff  |
| ----------------- | ------------- | ----------------- | ----- |
| ifeval            | 85.77         | 85.77             | 0     |
| hle               | 2.18          | 1.95              | -0.23 |
| gpqa              | 51.39         | 49.37             | -2.02 |
| aime2025          | 18.02         | 19.58             | 1.56  |
| mmlu_pro          | 74.05         | 72.86             | -1.19 |
| LCBCodeGeneration | 30.38         | 27.43             | -2.95 |

复现方式可以参考[这份](https://lmdeploy.readthedocs.io/zh-cn/latest/benchmark/evaluate_with_opencompass.html)文档
