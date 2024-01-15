<div align="center">
  <img src="docs/en/_static/image/lmdeploy-logo.svg" width="450"/>

[![docs](https://img.shields.io/badge/docs-latest-blue)](https://lmdeploy.readthedocs.io/en/latest/)
[![badge](https://github.com/InternLM/lmdeploy/workflows/lint/badge.svg)](https://github.com/InternLM/lmdeploy/actions)
[![PyPI](https://img.shields.io/pypi/v/lmdeploy)](https://pypi.org/project/lmdeploy)
[![license](https://img.shields.io/github/license/InternLM/lmdeploy.svg)](https://github.com/InternLM/lmdeploy/tree/main/LICENSE)
[![issue resolution](https://img.shields.io/github/issues-closed-raw/InternLM/lmdeploy)](https://github.com/InternLM/lmdeploy/issues)
[![open issues](https://img.shields.io/github/issues-raw/InternLM/lmdeploy)](https://github.com/InternLM/lmdeploy/issues)

English | [简体中文](README_zh-CN.md)

</div>

<p align="center">
    👋 join us on <a href="https://twitter.com/intern_lm" target="_blank">Twitter</a>, <a href="https://discord.gg/xa29JuW87d" target="_blank">Discord</a> and <a href="https://r.vansin.top/?r=internwx" target="_blank">WeChat</a>
</p>

______________________________________________________________________

## Latest News 🎉

- \[2024/01\] Support for multi-model, multi-machine, multi-card inference services. For usage instructions, please refer to[here](./docs/zh_cn/serving/proxy_server.md)
- \[2023/12\] Turbomind supports multimodal input. [Gradio Demo](./examples/vl/README.md)

______________________________________________________________________

# Introduction

LMDeploy is a toolkit for compressing, deploying, and serving LLM, developed by the [MMRazor](https://github.com/open-mmlab/mmrazor) and [MMDeploy](https://github.com/open-mmlab/mmdeploy) teams. It has the following core features:

- **Efficient Inference**: LMDeploy delivers up to 1.8x higher request throughput than vLLM, by introducing key features like persistent batch(a.k.a. continuous batching), blocked KV cache, dynamic split&fuse, tensor parallelism, high-performance CUDA kernels and so on.

- **Effective Quantization**: LMDeploy supports weight-only and k/v quantization, and the 4-bit inference performance is 2.4x higher than FP16. The quantization quality has been confirmed via OpenCompass evaluation.

- **Effortless Distribution Server**: Leveraging the request distribution service, LMDeploy facilitates an easy and efficient deployment of multi-model services across multiple machines and cards.

- **Interactive Inference Mode**: By caching the k/v of attention during multi-round dialogue processes, the engine remembers dialogue history, thus avoiding repetitive processing of historical sessions.

# Performance

![v0 1 0-benchmark](https://github.com/InternLM/lmdeploy/assets/4560679/8e455cf1-a792-4fa8-91a2-75df96a2a5ba)

For detailed inference benchmarks in more devices and more settings, please refer to the following link:

- [A100](./docs/en/benchmark/a100_fp16.md)
- V100
- 4090
- 3090
- 2080

# Supported Models

|       Model        |   Size    |
| :----------------: | :-------: |
|       Llama        | 7B - 65B  |
|       Llama2       | 7B - 70B  |
|      InternLM      | 7B - 20B  |
| InternLM-XComposer |    7B     |
|        QWen        | 7B - 72B  |
|      QWen-VL       |    7B     |
|      Baichuan      | 7B - 13B  |
|     Baichuan2      | 7B - 13B  |
|     Code Llama     | 7B - 34B  |
|      ChatGLM2      |    6B     |
|       Falcon       | 7B - 180B |

LMDeploy has developed two inference engines - [TurboMind](./docs/en/inference/turbomind.md) and [PyTorch](./docs/en/inference/pytorch.md), each with a different focus. The former strives for ultimate optimization of inference performance, while the latter, developed purely in Python, aims to decrease the barriers for developers.

They differ in the types of supported models and the inference data type. Please refer to [this table](./docs/en/supported_models/supported_models.md) for each engine's capability and choose the proper one that best fits your actual needs.

# Quick Start

## Installation

Install lmdeploy with pip ( python 3.8+) or [from source](./docs/en/build.md)

```shell
pip install lmdeploy
```

## Offline Batch Inference

```shell
import lmdeploy
pipe = lmdeploy.pipeline("internlm/internlm-chat-7b")
response = pipe(["Hi, pls intro yourself", "Shanghai is"])
print(response)
```

For more information about inference pipeline, please refer to [here](./docs/en/inference/pipeline.md).

# Tutorials

Please overview [getting_started](./docs/en/get_started.md) section for the basic usage of LMDeploy.

For detailed user guides and advanced guides, please refer to our [tutorials](https://lmdeploy.readthedocs.io/en/latest/):

- User Guide
  - [Inference pipeline](./docs/en/inference/pipeline.md)
  - [Inference Engine - TurboMind](docs/en/inference/turbomind.md)
  - [Inference Engine - PyTorch](docs/en/inference/pytorch.md)
  - [Serving](docs/en/serving/restful_api.md)
  - [Quantization](docs/en/quantization)
- Advance Guide
  - Add chat template
  - Add a new model
  - gemm tuning
  - Long context inference
  - [Multi-model inference service](docs/en/serving/proxy_server.md)

## Contributing

We appreciate all contributions to LMDeploy. Please refer to [CONTRIBUTING.md](.github/CONTRIBUTING.md) for the contributing guideline.

## Acknowledgement

- [FasterTransformer](https://github.com/NVIDIA/FasterTransformer)
- [llm-awq](https://github.com/mit-han-lab/llm-awq)
- [vLLM](https://github.com/vllm-project/vllm)
- [DeepSpeed-MII](https://github.com/microsoft/DeepSpeed-MII)

## License

This project is released under the [Apache 2.0 license](LICENSE).
