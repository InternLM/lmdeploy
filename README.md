<div align="center">
  <img src="resources/lmdeploy-logo.svg" width="450"/>

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

- \[2023/12\] Turbomind supports multimodal input. [Gradio Demo](./examples/vl/README.md)
- \[2023/11\] Turbomind supports loading hf model directly. Click [here](./docs/en/load_hf.md) for details.
- \[2023/11\] TurboMind major upgrades, including: Paged Attention, faster attention kernels without sequence length limitation, 2x faster KV8 kernels, Split-K decoding (Flash Decoding), and W4A16 inference for sm_75
- \[2023/09\] TurboMind supports Qwen-14B
- \[2023/09\] TurboMind supports InternLM-20B
- \[2023/09\] TurboMind supports all features of Code Llama: code completion, infilling, chat / instruct, and python specialist. Click [here](./docs/en/supported_models/codellama.md) for deployment guide
- \[2023/09\] TurboMind supports Baichuan2-7B
- \[2023/08\] TurboMind supports flash-attention2.
- \[2023/08\] TurboMind supports Qwen-7B, dynamic NTK-RoPE scaling and dynamic logN scaling
- \[2023/08\] TurboMind supports Windows (tp=1)
- \[2023/08\] TurboMind supports 4-bit inference, 2.4x faster than FP16, the fastest open-source implementation🚀. Check [this](./docs/en/w4a16.md) guide for detailed info
- \[2023/08\] LMDeploy has launched on the [HuggingFace Hub](https://huggingface.co/lmdeploy), providing ready-to-use 4-bit models.
- \[2023/08\] LMDeploy supports 4-bit quantization using the [AWQ](https://arxiv.org/abs/2306.00978) algorithm.
- \[2023/07\] TurboMind supports Llama-2 70B with GQA.
- \[2023/07\] TurboMind supports Llama-2 7B/13B.
- \[2023/07\] TurboMind supports tensor-parallel inference of InternLM.

______________________________________________________________________

# Introduction

LMDeploy is a toolkit for compressing, deploying, and serving LLM, developed by the [MMRazor](https://github.com/open-mmlab/mmrazor) and [MMDeploy](https://github.com/open-mmlab/mmdeploy) teams. It has the following core features:

- **Efficient Inference Engine (TurboMind)**: Supports  several features such as blocked KV-caching, continuous batching, Dynamic SplitFuse, tensor parallelism, and high-performance CUDA kernels to support fast high throughput text-generation for LLMs such as Llama-2-70B. MII delivers up to 2.3 times higher effective throughput compared to leading systems such as vLLM.

- **Interactive Inference Mode**: By caching the k/v of attention during multi-round dialogue processes, it remembers dialogue history, thus avoiding repetitive processing of historical sessions.

- **Multi-GPU Model Deployment and Quantization**: We provide comprehensive model deployment and quantification support, and have been validated at different scales.

- **Persistent Batch Inference**: Further optimization of model execution efficiency.

# Performance

The TurboMind engine achieves up to 1.36 ~ 1.85 times higher request throughput compared to vLLM across models of various size. In terms of static inference capabilities, the token throughput (`out token/s`) of TurboMind's 4bit model inference significantly outperforms FP16/BF16 inference, with an improvement of up to 2.4 times.

![v0 1 0-benchmark](https://github.com/InternLM/lmdeploy/assets/4560679/f4d218f9-db3b-4ceb-ab50-97cb005b3ac9)

# Supported Models

`LMDeploy` has developed two inference engines - `Pytorch` and `TurboMind`, each with different emphases. The former strives for ultimate optimization of inference performance, while the latter, developed purely in Python, aims to decrease the barriers for developers.

As shown in the next tables, the inference engines differ in the types of supported models and the inference data type. Users can choose the one that best fits their actual needs.

## TurboMind

|       Model        |   Size   | FP16/BF16 | KV INT8 | W4A16 |
| :----------------: | :------: | :-------: | :-----: | :---: |
|       Llama        | 7B - 65B |    Yes    |   Yes   |  Yes  |
|       Llama2       | 7B - 70B |    Yes    |   Yes   |  Yes  |
|      InternLM      | 7B - 20B |    Yes    |   Yes   |  Yes  |
| InternLM-XComposer |    7B    |    Yes    |   Yes   |  Yes  |
|        QWen        | 7B - 72B |    Yes    |   Yes   |  Yes  |
|      QWen-VL       |    7B    |    Yes    |   Yes   |  Yes  |
|      Baichuan      |    7B    |    Yes    |   Yes   |  Yes  |
|     Baichuan2      |    7B    |    Yes    |   Yes   |  Yes  |
|     Code Llama     | 7B - 34B |    Yes    |   No    |  No   |

## Pytorch

|   Model   |   Size    | FP16/BF16 | KV INT8 | W8A8 |
| :-------: | :-------: | :-------: | :-----: | :--: |
|   Llama   | 7B - 65B  |    Yes    |   No    | Yes  |
|  Llama2   | 7B - 70B  |    Yes    |   No    | Yes  |
| InternLM  | 7B - 20B  |    Yes    |   No    | Yes  |
| Baichuan2 | 7B - 13B  |    Yes    |   No    | Yes  |
| ChatGLM2  |    6B     |    Yes    |   No    |  No  |
|  Falcon   | 7B - 180B |    Yes    |   No    |  No  |

# Quick Start

LMDeploy offers functionalities such as model quantization, offline batch inference, online serving, etc. Each function can be completed with just a few simple lines of code or commands.

<!-- toc -->

- [Installation](#Installation)
- [Offline Batch Inference](#offline-batch-inference)
- [Serving](#serving)
- [Quantization](#quantization)
- [Utilities](#utilities)

<!-- tocstop -->

## Installation

Install lmdeploy with pip ( python 3.8+) or [from source](./docs/en/build.md)

```shell
pip install lmdeploy
```

## Offline batch inference

```shell
import lmdeploy
pipe = lmdeploy.pipeline("internlm/internlm-chat-7b", tp=1)
response = pipe(["Hi, pls intro yourself", "Shanghai is"])
print(response)
```

Tensor parallelism is supported, and can be invoked by setting the `tp` parameter. For more information on inference pipeline parameters, please refer to [here(TODO)](<>).

## Serving

LMDeploy's `api_server` allows for one-click encapsulation of models into services. The provided RESTful API is compatible with OpenAI's interface. Below are examples of service startup and request handling:

```shell
# launch api_server
lmdeploy serve api_server internlm/internlm-chat-7b --server-port 8080 --tp 1
# send request to api_server and receive the server's response
lmdeploy serve api_client http://0.0.0.0:8080
```

在上述例子中，服务启动后，在浏览器输入 `http://0.0.0.0:8080`，可在线阅读和试用 `api_server` 的各接口，也可直接查阅[文档](./docs/en/restful_api.md)，了解各接口的定义和使用方法。
After launching the server, users can overview and try out `api_server` APIs online by entering `http://0.0.0.0:8080`  in the browser. Besides,

## Quantization

#### Weight INT4 Quantization

LMDeploy uses [AWQ](https://arxiv.org/abs/2306.00978) algorithm for model weight quantization

只用两行命令，就可以把一个 LLM 模型权重量化为 4bit，并在控制台与模型进行交互式对话。

```shell
lmdeploy lite auto_awq internlm/internlm-chat-7b --work-dir ./internlm-chat-7b-4bit
lmdeploy chat turbomind ./internlm-chat-7b-4bit --model-format awq --group-size 128
```

LMDeploy 4bit 量化和推理支持的显卡包括：

- 图灵架构（sm75）：20系列、T4
- 安培架构（sm80,sm86）：30系列、A10、A16、A30、A100
- Ada Lovelace架构（sm90）：40 系列

[Click here](./docs/en/w4a16.md) to view the test results for weight int4 usage.

#### KV Cache INT8 Quantization

[Click here](./docs/en/kv_int8.md) to view the usage method, implementation formula, and test results for kv int8.

> **Warning**<br />
> runtime Tensor Parallel for quantized model is not available. Please setup `--tp` on `deploy` to enable static TP.

## Contributing

We appreciate all contributions to LMDeploy. Please refer to [CONTRIBUTING.md](.github/CONTRIBUTING.md) for the contributing guideline.

## Acknowledgement

- [FasterTransformer](https://github.com/NVIDIA/FasterTransformer)
- [llm-awq](https://github.com/mit-han-lab/llm-awq)

## License

This project is released under the [Apache 2.0 license](LICENSE).
