<div align="center">
  <img src="docs/en/_static/image/lmdeploy-logo.svg" width="450"/>

[![PyPI](https://img.shields.io/pypi/v/lmdeploy)](https://pypi.org/project/lmdeploy)
![PyPI - Downloads](https://img.shields.io/pypi/dm/lmdeploy)
[![license](https://img.shields.io/github/license/InternLM/lmdeploy.svg)](https://github.com/InternLM/lmdeploy/tree/main/LICENSE)
[![issue resolution](https://img.shields.io/github/issues-closed-raw/InternLM/lmdeploy)](https://github.com/InternLM/lmdeploy/issues)
[![open issues](https://img.shields.io/github/issues-raw/InternLM/lmdeploy)](https://github.com/InternLM/lmdeploy/issues)

[📘Documentation](https://lmdeploy.readthedocs.io/en/latest/) |
[🛠️Quick Start](https://lmdeploy.readthedocs.io/en/latest/get_started.html) |
[🤔Reporting Issues](https://github.com/InternLM/lmdeploy/issues/new/choose)

English | [简体中文](README_zh-CN.md)

👋 join us on [![Static Badge](https://img.shields.io/badge/-grey?style=social&logo=wechat&label=WeChat)](https://r.vansin.top/?r=internwx)
[![Static Badge](https://img.shields.io/badge/-grey?style=social&logo=twitter&label=Twitter)](https://twitter.com/intern_lm)
[![Static Badge](https://img.shields.io/badge/-grey?style=social&logo=discord&label=Discord)](https://discord.gg/xa29JuW87d)

</div>

______________________________________________________________________

## Latest News 🎉

<details open>
<summary><b>2024</b></summary>

- \[2024/04\] Support Llama3 and more VLMs, such as InternVL v1.1, v1.2, MiniGemini, InternLMXComposer2.
- \[2024/04\] TurboMind adds online int8/int4 KV cache quantization and inference for all supported devices. Refer [here](docs/en/quantization/kv_quant.md) for detailed guide
- \[2024/04\] TurboMind latest upgrade boosts GQA, rocketing the [internlm2-20b](https://huggingface.co/internlm/internlm2-20b) model inference to 16+ RPS, about 1.8x faster than vLLM.
- \[2024/04\] Support Qwen1.5-MOE and dbrx.
- \[2024/03\] Support DeepSeek-VL offline inference pipeline and serving.
- \[2024/03\] Support VLM offline inference pipeline and serving.
- \[2024/02\] Support Qwen 1.5, Gemma, Mistral, Mixtral, Deepseek-MOE and so on.
- \[2024/01\] [OpenAOE](https://github.com/InternLM/OpenAOE) seamless integration with [LMDeploy Serving Service](./docs/en/serving/api_server.md).
- \[2024/01\] Support for multi-model, multi-machine, multi-card inference services. For usage instructions, please refer to [here](./docs/en/serving/proxy_server.md)
- \[2024/01\] Support [PyTorch inference engine](./docs/en/inference/pytorch.md), developed entirely in Python, helping to lower the barriers for developers and enable  rapid experimentation with new features and technologies.

</details>

<details close>
<summary><b>2023</b></summary>

- \[2023/12\] Turbomind supports multimodal input. [Gradio Demo](./examples/vl/README.md)
- \[2023/11\] Turbomind supports loading hf model directly. Click [here](docs/en/inference/load_hf.md) for details.
- \[2023/11\] TurboMind major upgrades, including: Paged Attention, faster attention kernels without sequence length limitation, 2x faster KV8 kernels, Split-K decoding (Flash Decoding), and W4A16 inference for sm_75
- \[2023/09\] TurboMind supports Qwen-14B
- \[2023/09\] TurboMind supports InternLM-20B
- \[2023/09\] TurboMind supports all features of Code Llama: code completion, infilling, chat / instruct, and python specialist. Click [here](./docs/en/supported_models/codellama.md) for deployment guide
- \[2023/09\] TurboMind supports Baichuan2-7B
- \[2023/08\] TurboMind supports flash-attention2.
- \[2023/08\] TurboMind supports Qwen-7B, dynamic NTK-RoPE scaling and dynamic logN scaling
- \[2023/08\] TurboMind supports Windows (tp=1)
- \[2023/08\] TurboMind supports 4-bit inference, 2.4x faster than FP16, the fastest open-source implementation. Check [this](docs/en/quantization/w4a16.md) guide for detailed info
- \[2023/08\] LMDeploy has launched on the [HuggingFace Hub](https://huggingface.co/lmdeploy), providing ready-to-use 4-bit models.
- \[2023/08\] LMDeploy supports 4-bit quantization using the [AWQ](https://arxiv.org/abs/2306.00978) algorithm.
- \[2023/07\] TurboMind supports Llama-2 70B with GQA.
- \[2023/07\] TurboMind supports Llama-2 7B/13B.
- \[2023/07\] TurboMind supports tensor-parallel inference of InternLM.

</details>

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

<table>
    <tr>
        <th colspan="2" align="center">LLM</th>
        <th colspan="2" align="center">VLM</th>
    </tr>
    <tr>
        <td>model</td>
        <td>size</td>
        <td>model</td>
        <td>size</td>
    </tr>
    <tr>
        <td>Llama</td>
        <td>7B - 65B</td>
        <td>LLaVA(1.5,1.6)</td>
        <td>7B - 34B</td>
    </tr>
    <tr>
        <td>Llama2</td>
        <td>7B - 70B</td>
        <td>QWen-VL</td>
        <td>7B</td>
    </tr>
    <tr>
        <td>Llama3</td>
        <td>8B, 70B</td>
        <td>InternLM-XComposer</td>
        <td>7B</td>
    </tr>
    <tr>
        <td>InternLM</td>
        <td>7B - 20B</td>
        <td>InternLM-XComposer2</td>
        <td>7B, 4khd-7B</td>
    </tr>
    <tr>
        <td>InternLM2</td>
        <td>7B - 20B</td>
        <td>DeepSeek-VL</td>
        <td>7B</td>
    </tr>
    <tr>
        <td>QWen</td>
        <td>1.8B - 72B</td>
        <td>InternVL-Chat</td>
        <td>-</td>
    </tr>
    <tr>
        <td>QWen1.5</td>
        <td>0.5B - 72B</td>
        <td>MiniGeminiLlama</td>
        <td>7B</td>
    </tr>
    <tr>
        <td>QWen1.5-MoE</td>
        <td>A2.7B</td>
        <td></td>
        <td></td>
    </tr>
    <tr>
        <td>Baichuan</td>
        <td>7B</td>
        <td></td>
        <td></td>
    </tr>
    <tr>
        <td>Baichuan2</td>
        <td>7B - 13B</td>
        <td></td>
        <td></td>
    </tr>
    <tr>
        <td>Code Llama</td>
        <td>7B - 34B</td>
        <td></td>
        <td></td>
    </tr>
    <tr>
        <td>ChatGLM2</td>
        <td>6B</td>
        <td></td>
        <td></td>
    </tr>
    <tr>
        <td>Falcon</td>
        <td>7B - 180B</td>
        <td></td>
        <td></td>
    </tr>
    <tr>
        <td>YI</td>
        <td>6B - 34B</td>
        <td></td>
        <td></td>
    </tr>
    <tr>
        <td>Mistral</td>
        <td>7B</td>
        <td></td>
        <td></td>
    </tr>
    <tr>
        <td>DeepSeek-MoE</td>
        <td>16B</td>
        <td></td>
        <td></td>
    </tr>
    <tr>
        <td>Mixtral</td>
        <td>8x7B</td>
        <td></td>
        <td></td>
    </tr>
    <tr>
        <td>Gemma</td>
        <td>2B-7B</td>
        <td></td>
        <td></td>
    </tr>
    <tr>
        <td>Dbrx</td>
        <td>132B</td>
    </tr>
</table>

LMDeploy has developed two inference engines - [TurboMind](./docs/en/inference/turbomind.md) and [PyTorch](./docs/en/inference/pytorch.md), each with a different focus. The former strives for ultimate optimization of inference performance, while the latter, developed purely in Python, aims to decrease the barriers for developers.

They differ in the types of supported models and the inference data type. Please refer to [this table](./docs/en/supported_models/supported_models.md) for each engine's capability and choose the proper one that best fits your actual needs.

# Quick Start [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1Dh-YlSwg78ZO3AlleO441NF_QP2shs95#scrollTo=YALmXnwCG1pQ)

## Installation

Install lmdeploy with pip ( python 3.8+) or [from source](./docs/en/build.md)

```shell
pip install lmdeploy
```

Since v0.3.0, The default prebuilt package is compiled on **CUDA 12**. However, if CUDA 11+ is required, you can install lmdeploy by:

```shell
export LMDEPLOY_VERSION=0.3.0
export PYTHON_VERSION=38
pip install https://github.com/InternLM/lmdeploy/releases/download/v${LMDEPLOY_VERSION}/lmdeploy-${LMDEPLOY_VERSION}+cu118-cp${PYTHON_VERSION}-cp${PYTHON_VERSION}-manylinux2014_x86_64.whl --extra-index-url https://download.pytorch.org/whl/cu118
```

## Offline Batch Inference

```python
import lmdeploy
pipe = lmdeploy.pipeline("internlm/internlm-chat-7b")
response = pipe(["Hi, pls intro yourself", "Shanghai is"])
print(response)
```

> \[!NOTE\]
> By default, LMDeploy downloads model from HuggingFace. If you would like to use models from ModelScope, please install ModelScope by `pip install modelscope` and set the environment variable:
>
> `export LMDEPLOY_USE_MODELSCOPE=True`

For more information about inference pipeline, please refer to [here](./docs/en/inference/pipeline.md).

# Tutorials

Please overview [getting_started](./docs/en/get_started.md) section for the basic usage of LMDeploy.

For detailed user guides and advanced guides, please refer to our [tutorials](https://lmdeploy.readthedocs.io/en/latest/):

- User Guide
  - [LLM Inference pipeline](./docs/en/inference/pipeline.md) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1Dh-YlSwg78ZO3AlleO441NF_QP2shs95#scrollTo=YALmXnwCG1pQ)
  - [VLM Inference pipeline](./docs/en/inference/vl_pipeline.md) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1nKLfnPeDA3p-FMNw2NhI-KOpk7-nlNjF?usp=sharing)
  - [LLM Serving](docs/en/serving/api_server.md)
  - [VLM Serving](docs/en/serving/api_server_vl.md)
  - [Quantization](docs/en/quantization)
- Advance Guide
  - [Inference Engine - TurboMind](docs/en/inference/turbomind.md)
  - [Inference Engine - PyTorch](docs/en/inference/pytorch.md)
  - [Customize chat templates](docs/en/advance/chat_template.md)
  - [Add a new model](docs/en/advance/pytorch_new_model.md)
  - gemm tuning
  - [Long context inference](docs/en/advance/long_context.md)
  - [Multi-model inference service](docs/en/serving/proxy_server.md)

# Third-party projects

- Deploying LLMs offline on the NVIDIA Jetson platform by LMDeploy: [LMDeploy-Jetson](https://github.com/BestAnHongjun/LMDeploy-Jetson)

# Contributing

We appreciate all contributions to LMDeploy. Please refer to [CONTRIBUTING.md](.github/CONTRIBUTING.md) for the contributing guideline.

# Acknowledgement

- [FasterTransformer](https://github.com/NVIDIA/FasterTransformer)
- [llm-awq](https://github.com/mit-han-lab/llm-awq)
- [vLLM](https://github.com/vllm-project/vllm)
- [DeepSpeed-MII](https://github.com/microsoft/DeepSpeed-MII)

# Citation

```bibtex
@misc{2023lmdeploy,
    title={LMDeploy: A Toolkit for Compressing, Deploying, and Serving LLM},
    author={LMDeploy Contributors},
    howpublished = {\url{https://github.com/InternLM/lmdeploy}},
    year={2023}
}
```

# License

This project is released under the [Apache 2.0 license](LICENSE).
