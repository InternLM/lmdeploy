<div align="center">
  <img src="docs/en/_static/image/lmdeploy-logo.svg" width="450"/>

[![PyPI](https://img.shields.io/pypi/v/lmdeploy)](https://pypi.org/project/lmdeploy)
![PyPI - Downloads](https://img.shields.io/pypi/dm/lmdeploy)
[![license](https://img.shields.io/github/license/InternLM/lmdeploy.svg)](https://github.com/InternLM/lmdeploy/tree/main/LICENSE)
[![issue resolution](https://img.shields.io/github/issues-closed-raw/InternLM/lmdeploy)](https://github.com/InternLM/lmdeploy/issues)
[![open issues](https://img.shields.io/github/issues-raw/InternLM/lmdeploy)](https://github.com/InternLM/lmdeploy/issues)

[📘Documentation](https://lmdeploy.readthedocs.io/en/latest/) |
[🛠️Quick Start](https://lmdeploy.readthedocs.io/en/latest/get_started/get_started.html) |
[🤔Reporting Issues](https://github.com/InternLM/lmdeploy/issues/new/choose)

English | [简体中文](README_zh-CN.md) | [日本語](README_ja.md)

👋 join us on [![Static Badge](https://img.shields.io/badge/-grey?style=social&logo=wechat&label=WeChat)](https://cdn.vansin.top/internlm/lmdeploy.jpg)
[![Static Badge](https://img.shields.io/badge/-grey?style=social&logo=twitter&label=Twitter)](https://twitter.com/intern_lm)
[![Static Badge](https://img.shields.io/badge/-grey?style=social&logo=discord&label=Discord)](https://discord.gg/xa29JuW87d)

</div>

______________________________________________________________________

## Latest News 🎉

<details open>
<summary><b>2025</b></summary>

- \[2025/09\] TurboMind supports MXFP4 on NVIDIA GPUs starting from V100, achieving 1.5x the performmance of vLLM on H800 for openai gpt-oss models!
- \[2025/06\] Comprehensive inference optimization for FP8 MoE Models
- \[2025/06\] DeepSeek PD Disaggregation deployment is now supported through integration with [DLSlime](https://github.com/DeepLink-org/DLSlime) and [Mooncake](https://github.com/kvcache-ai/Mooncake). Huge thanks to both teams!
- \[2025/04\] Enhance DeepSeek inference performance by integration deepseek-ai techniques: FlashMLA, DeepGemm, DeepEP, MicroBatch and eplb
- \[2025/01\] Support DeepSeek V3 and R1

</details>

<details close>
<summary><b>2024</b></summary>

- \[2024/11\] Support Mono-InternVL with PyTorch engine
- \[2024/10\] PyTorchEngine supports graph mode on ascend platform, doubling the inference speed
- \[2024/09\] LMDeploy PyTorchEngine adds support for [Huawei Ascend](./docs/en/get_started/ascend/get_started.md). See supported models [here](docs/en/supported_models/supported_models.md)
- \[2024/09\] LMDeploy PyTorchEngine achieves 1.3x faster on Llama3-8B inference by introducing CUDA graph
- \[2024/08\] LMDeploy is integrated into [modelscope/swift](https://github.com/modelscope/swift) as the default accelerator for VLMs inference
- \[2024/07\] Support Llama3.1 8B, 70B and its TOOLS CALLING
- \[2024/07\] Support [InternVL2](docs/en/multi_modal/internvl.md) full-series models, [InternLM-XComposer2.5](docs/en/multi_modal/xcomposer2d5.md) and [function call](docs/en/llm/api_server_tools.md) of InternLM2.5
- \[2024/06\] PyTorch engine support DeepSeek-V2 and several VLMs, such as CogVLM2, Mini-InternVL, LlaVA-Next
- \[2024/05\] Balance vision model when deploying VLMs with multiple GPUs
- \[2024/05\] Support 4-bits weight-only quantization and inference on VLMs, such as InternVL v1.5, LLaVa, InternLMXComposer2
- \[2024/04\] Support Llama3 and more VLMs, such as InternVL v1.1, v1.2, MiniGemini, InternLMXComposer2.
- \[2024/04\] TurboMind adds online int8/int4 KV cache quantization and inference for all supported devices. Refer [here](docs/en/quantization/kv_quant.md) for detailed guide
- \[2024/04\] TurboMind latest upgrade boosts GQA, rocketing the [internlm2-20b](https://huggingface.co/internlm/internlm2-20b) model inference to 16+ RPS, about 1.8x faster than vLLM.
- \[2024/04\] Support Qwen1.5-MOE and dbrx.
- \[2024/03\] Support DeepSeek-VL offline inference pipeline and serving.
- \[2024/03\] Support VLM offline inference pipeline and serving.
- \[2024/02\] Support Qwen 1.5, Gemma, Mistral, Mixtral, Deepseek-MOE and so on.
- \[2024/01\] [OpenAOE](https://github.com/InternLM/OpenAOE) seamless integration with [LMDeploy Serving Service](docs/en/llm/api_server.md).
- \[2024/01\] Support for multi-model, multi-machine, multi-card inference services. For usage instructions, please refer to [here](docs/en/llm/proxy_server.md)
- \[2024/01\] Support [PyTorch inference engine](./docs/en/inference/pytorch.md), developed entirely in Python, helping to lower the barriers for developers and enable  rapid experimentation with new features and technologies.

</details>

<details close>
<summary><b>2023</b></summary>

- \[2023/12\] Turbomind supports multimodal input.
- \[2023/11\] Turbomind supports loading hf model directly. Click [here](docs/en/inference/load_hf.md) for details.
- \[2023/11\] TurboMind major upgrades, including: Paged Attention, faster attention kernels without sequence length limitation, 2x faster KV8 kernels, Split-K decoding (Flash Decoding), and W4A16 inference for sm_75
- \[2023/09\] TurboMind supports Qwen-14B
- \[2023/09\] TurboMind supports InternLM-20B
- \[2023/09\] TurboMind supports all features of Code Llama: code completion, infilling, chat / instruct, and python specialist. Click [here](./docs/en/llm/codellama.md) for deployment guide
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

- **Excellent Compatibility**: LMDeploy supports [KV Cache Quant](docs/en/quantization/kv_quant.md), [AWQ](docs/en/quantization/w4a16.md) and [Automatic Prefix Caching](docs/en/inference/turbomind_config.md) to be used simultaneously.

# Performance

![v0 1 0-benchmark](https://github.com/InternLM/lmdeploy/assets/4560679/8e455cf1-a792-4fa8-91a2-75df96a2a5ba)

# Supported Models

<table>
<tbody>
<tr align="center" valign="middle">
<td>
  <b>LLMs</b>
</td>
<td>
  <b>VLMs</b>
</td>
<tr valign="top">
<td align="left" valign="top">
<ul>
  <li>Llama (7B - 65B)</li>
  <li>Llama2 (7B - 70B)</li>
  <li>Llama3 (8B, 70B)</li>
  <li>Llama3.1 (8B, 70B)</li>
  <li>Llama3.2 (1B, 3B)</li>
  <li>InternLM (7B - 20B)</li>
  <li>InternLM2 (7B - 20B)</li>
  <li>InternLM3 (8B)</li>
  <li>InternLM2.5 (7B)</li>
  <li>Qwen (1.8B - 72B)</li>
  <li>Qwen1.5 (0.5B - 110B)</li>
  <li>Qwen1.5 - MoE (0.5B - 72B)</li>
  <li>Qwen2 (0.5B - 72B)</li>
  <li>Qwen2-MoE (57BA14B)</li>
  <li>Qwen2.5 (0.5B - 32B)</li>
  <li>Qwen3, Qwen3-MoE</li>
  <li>Qwen3-Next(80B)</li>
  <li>Baichuan (7B)</li>
  <li>Baichuan2 (7B-13B)</li>
  <li>Code Llama (7B - 34B)</li>
  <li>ChatGLM2 (6B)</li>
  <li>GLM-4 (9B)</li>
  <li>GLM-4-0414 (9B, 32B)</li>
  <li>CodeGeeX4 (9B)</li>
  <li>YI (6B-34B)</li>
  <li>Mistral (7B)</li>
  <li>DeepSeek-MoE (16B)</li>
  <li>DeepSeek-V2 (16B, 236B)</li>
  <li>DeepSeek-V2.5 (236B)</li>
  <li>Mixtral (8x7B, 8x22B)</li>
  <li>Gemma (2B - 7B)</li>
  <li>StarCoder2 (3B - 15B)</li>
  <li>Phi-3-mini (3.8B)</li>
  <li>Phi-3.5-mini (3.8B)</li>
  <li>Phi-3.5-MoE (16x3.8B)</li>
  <li>Phi-4-mini (3.8B)</li>
  <li>MiniCPM3 (4B)</li>
  <li>SDAR (1.7B-30B)</li>
  <li>gpt-oss (20B, 120B)</li>
</ul>
</td>
<td>
<ul>
  <li>LLaVA(1.5,1.6) (7B-34B)</li>
  <li>InternLM-XComposer2 (7B, 4khd-7B)</li>
  <li>InternLM-XComposer2.5 (7B)</li>
  <li>Qwen-VL (7B)</li>
  <li>Qwen2-VL (2B, 7B, 72B)</li>
  <li>Qwen2.5-VL (3B, 7B, 72B)</li>
  <li>Qwen3-VL (2B - 235B)</li>
  <li>DeepSeek-VL (7B)</li>
  <li>DeepSeek-VL2 (3B, 16B, 27B)</li>
  <li>InternVL-Chat (v1.1-v1.5)</li>
  <li>InternVL2 (1B-76B)</li>
  <li>InternVL2.5(MPO) (1B-78B)</li>
  <li>InternVL3 (1B-78B)</li>
  <li>InternVL3.5 (1B-241BA28B)</li>
  <li>Intern-S1 (241B)</li>
  <li>Intern-S1-mini (8.3B)</li>
  <li>Mono-InternVL (2B)</li>
  <li>ChemVLM (8B-26B)</li>
  <li>CogVLM-Chat (17B)</li>
  <li>CogVLM2-Chat (19B)</li>
  <li>MiniCPM-Llama3-V-2_5</li>
  <li>MiniCPM-V-2_6</li>
  <li>Phi-3-vision (4.2B)</li>
  <li>Phi-3.5-vision (4.2B)</li>
  <li>GLM-4V (9B)</li>
  <li>GLM-4.1V-Thinking (9B)</li>
  <li>Llama3.2-vision (11B, 90B)</li>
  <li>Molmo (7B-D,72B)</li>
  <li>Gemma3 (1B - 27B)</li>
  <li>Llama4 (Scout, Maverick)</li>
</ul>
</td>
</tr>
</tbody>
</table>

LMDeploy has developed two inference engines - [TurboMind](./docs/en/inference/turbomind.md) and [PyTorch](./docs/en/inference/pytorch.md), each with a different focus. The former strives for ultimate optimization of inference performance, while the latter, developed purely in Python, aims to decrease the barriers for developers.

They differ in the types of supported models and the inference data type. Please refer to [this table](./docs/en/supported_models/supported_models.md) for each engine's capability and choose the proper one that best fits your actual needs.

# Quick Start [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1Dh-YlSwg78ZO3AlleO441NF_QP2shs95#scrollTo=YALmXnwCG1pQ)

## Installation

It is recommended installing lmdeploy using pip in a conda environment (python 3.9 - 3.13):

```shell
conda create -n lmdeploy python=3.10 -y
conda activate lmdeploy
pip install lmdeploy
```

The default prebuilt package is compiled on **CUDA 12** since v0.3.0.

For the GeForce RTX 50 series, please install the LMDeploy prebuilt package complied with **CUDA 12.8**

```shell
export LMDEPLOY_VERSION=0.10.2
export PYTHON_VERSION=310
pip install https://github.com/InternLM/lmdeploy/releases/download/v${LMDEPLOY_VERSION}/lmdeploy-${LMDEPLOY_VERSION}+cu128-cp${PYTHON_VERSION}-cp${PYTHON_VERSION}-manylinux2014_x86_64.whl --extra-index-url https://download.pytorch.org/whl/cu128
```

For more information on installing on CUDA 11+ platform, or for instructions on building from source, please refer to the [installation guide](docs/en/get_started/installation.md).

## Offline Batch Inference

```python
import lmdeploy
with lmdeploy.pipeline("internlm/internlm3-8b-instruct") as pipe:
    response = pipe(["Hi, pls intro yourself", "Shanghai is"])
    print(response)
```

> \[!NOTE\]
> By default, LMDeploy downloads model from HuggingFace. If you would like to use models from ModelScope, please install ModelScope by `pip install modelscope` and set the environment variable:
>
> `export LMDEPLOY_USE_MODELSCOPE=True`
>
> If you would like to use models from openMind Hub, please install openMind Hub by `pip install openmind_hub` and set the environment variable:
>
> `export LMDEPLOY_USE_OPENMIND_HUB=True`

For more information about inference pipeline, please refer to [here](docs/en/llm/pipeline.md).

# Tutorials

Please review [getting_started](docs/en/get_started/get_started.md) section for the basic usage of LMDeploy.

For detailed user guides and advanced guides, please refer to our [tutorials](https://lmdeploy.readthedocs.io/en/latest/):

- User Guide
  - [LLM Inference pipeline](docs/en/llm/pipeline.md) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1Dh-YlSwg78ZO3AlleO441NF_QP2shs95#scrollTo=YALmXnwCG1pQ)
  - [VLM Inference pipeline](docs/en/multi_modal/vl_pipeline.md) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1nKLfnPeDA3p-FMNw2NhI-KOpk7-nlNjF?usp=sharing)
  - [LLM Serving](docs/en/llm/api_server.md)
  - [VLM Serving](docs/en/multi_modal/api_server_vl.md)
  - [Quantization](docs/en/quantization)
- Advance Guide
  - [Inference Engine - TurboMind](docs/en/inference/turbomind.md)
  - [Inference Engine - PyTorch](docs/en/inference/pytorch.md)
  - [Customize chat templates](docs/en/advance/chat_template.md)
  - [Add a new model](docs/en/advance/pytorch_new_model.md)
  - gemm tuning
  - [Long context inference](docs/en/advance/long_context.md)
  - [Multi-model inference service](docs/en/llm/proxy_server.md)

# Third-party projects

- Deploying LLMs offline on the NVIDIA Jetson platform by LMDeploy: [LMDeploy-Jetson](https://github.com/BestAnHongjun/LMDeploy-Jetson)

- Example project for deploying LLMs using LMDeploy and BentoML: [BentoLMDeploy](https://github.com/bentoml/BentoLMDeploy)

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

```bibtex
@article{zhang2025efficient,
  title={Efficient Mixed-Precision Large Language Model Inference with TurboMind},
  author={Zhang, Li and Jiang, Youhe and He, Guoliang and Chen, Xin and Lv, Han and Yao, Qian and Fu, Fangcheng and Chen, Kai},
  journal={arXiv preprint arXiv:2508.15601},
  year={2025}
}
```

# License

This project is released under the [Apache 2.0 license](LICENSE).
