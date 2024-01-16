<div align="center">
  <img src="docs/en/_static/image/lmdeploy-logo.svg" width="450"/>

[![docs](https://img.shields.io/badge/docs-latest-blue)](https://lmdeploy.readthedocs.io/zh-cn/latest/)
[![badge](https://github.com/InternLM/lmdeploy/workflows/lint/badge.svg)](https://github.com/InternLM/lmdeploy/actions)
[![PyPI](https://img.shields.io/pypi/v/lmdeploy)](https://pypi.org/project/lmdeploy)
[![license](https://img.shields.io/github/license/InternLM/lmdeploy.svg)](https://github.com/InternLM/lmdeploy/tree/main/LICENSE)
[![issue resolution](https://img.shields.io/github/issues-closed-raw/InternLM/lmdeploy)](https://github.com/InternLM/lmdeploy/issues)
[![open issues](https://img.shields.io/github/issues-raw/InternLM/lmdeploy)](https://github.com/InternLM/lmdeploy/issues)

[English](README.md) | 简体中文

</div>

<p align="center">
    👋 join us on <a href="https://twitter.com/intern_lm" target="_blank">Twitter</a>, <a href="https://discord.gg/xa29JuW87d" target="_blank">Discord</a> and <a href="https://r.vansin.top/?r=internwx" target="_blank">WeChat</a>
</p>

______________________________________________________________________

## 最新进展 🎉

<details open>
<summary><b>2024</b></summary>

- \[2024/01\] 支持多模型、多机、多卡推理服务。使用方法请参考[此处](./docs/zh_cn/serving/proxy_server.md)
- \[2024/01\] 增加 [PyTorch 推理引擎](./docs/zh_cn/inference/pytorch.md)，作为 TurboMind 引擎的补充。帮助降低开发门槛，和快速实验新特性、新技术

</details>

<details close>
<summary><b>2023</b></summary>

- \[2023/12\] Turbomind 支持多模态输入。[Gradio Demo](./examples/vl/README.md)
- \[2023/11\] Turbomind 支持直接读取 Huggingface 模型。点击[这里](docs/zh_cn/inference/load_hf.md)查看使用方法
- \[2023/11\] TurboMind 重磅升级。包括：Paged Attention、更快的且不受序列最大长度限制的 attention kernel、2+倍快的 KV8 kernels、Split-K decoding (Flash Decoding) 和 支持 sm_75 架构的 W4A16
- \[2023/09\] TurboMind 支持 Qwen-14B
- \[2023/09\] TurboMind 支持 InternLM-20B 模型
- \[2023/09\] TurboMind 支持 Code Llama 所有功能：代码续写、填空、对话、Python专项。点击[这里](./docs/zh_cn/supported_models/codellama.md)阅读部署方法
- \[2023/09\] TurboMind 支持 Baichuan2-7B
- \[2023/08\] TurboMind 支持 flash-attention2
- \[2023/08\] TurboMind 支持 Qwen-7B，动态NTK-RoPE缩放，动态logN缩放
- \[2023/08\] TurboMind 支持 Windows (tp=1)
- \[2023/08\] TurboMind 支持 4-bit 推理，速度是 FP16 的 2.4 倍，是目前最快的开源实现。部署方式请看[这里](docs/zh_cn/quantization/w4a16.md)
- \[2023/08\] LMDeploy 开通了 [HuggingFace Hub](https://huggingface.co/lmdeploy) ，提供开箱即用的 4-bit 模型
- \[2023/08\] LMDeploy 支持使用 [AWQ](https://arxiv.org/abs/2306.00978) 算法进行 4-bit 量化
- \[2023/07\] TurboMind 支持使用 GQA 的 Llama-2 70B 模型
- \[2023/07\] TurboMind 支持 Llama-2 7B/13B 模型
- \[2023/07\] TurboMind 支持 InternLM 的 Tensor Parallel 推理

</details>
______________________________________________________________________

# 简介

LMDeploy 由 [MMDeploy](https://github.com/open-mmlab/mmdeploy) 和 [MMRazor](https://github.com/open-mmlab/mmrazor) 团队联合开发，是涵盖了 LLM 任务的全套轻量化、部署和服务解决方案。
这个强大的工具箱提供以下核心功能：

- **高效的推理**：LMDeploy 开发了 Persistent Batch(即 Continuous Batch)，Blocked K/V Cache，动态拆分和融合，张量并行，高效的计算 kernel等重要特性。推理性能是 vLLM 的 1.8 倍

- **可靠的量化**：LMDeploy 支持权重量化和 k/v 量化。4bit 模型推理效率是 FP16 下的 2.4 倍。量化模型的可靠性已通过 OpenCompass 评测得到充分验证。

- **便捷的服务**：通过请求分发服务，LMDeploy 支持多模型在多机、多卡上的推理服务。

- **有状态推理**：通过缓存多轮对话过程中 attention 的 k/v，记住对话历史，从而避免重复处理历史会话。显著提升长文本多轮对话场景中的效率。

# 性能

LMDeploy TurboMind 引擎拥有卓越的推理能力，在各种规模的模型上，每秒处理的请求数是 vLLM 的 1.36 ~ 1.85 倍。在静态推理能力方面，TurboMind 4bit 模型推理速度（out token/s）远高于 FP16/BF16 推理。在小 batch 时，提高到 2.4 倍。

![v0 1 0-benchmark](https://github.com/InternLM/lmdeploy/assets/4560679/8e455cf1-a792-4fa8-91a2-75df96a2a5ba)

更多设备、更多计算精度、更多setting下的的推理 benchmark，请参考以下链接：

- [A100](./docs/en/benchmark/a100_fp16.md)
- 4090
- 3090
- 2080

# 支持的模型

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

LMDeploy 支持 2 种推理引擎： [TurboMind](./docs/zh_cn/inference/turbomind.md) 和 [PyTorch](./docs/zh_cn/inference/pytorch.md)，它们侧重不同。前者追求推理性能的极致优化，后者纯用python开发，着重降低开发者的门槛。

它们在支持的模型类别、计算精度方面有所差别。用户可参考[这里](./docs/zh_cn/supported_models/supported_models.md), 查阅每个推理引擎的能力，并根据实际需求选择合适的。

# 快速开始

## 安装

使用 pip ( python 3.8+) 安装 LMDeploy，或者[源码安装](./docs/zh_cn/build.md)

```shell
pip install lmdeploy
```

## 离线批处理

```python
import lmdeploy
pipe = lmdeploy.pipeline("internlm/internlm-chat-7b")
response = pipe(["Hi, pls intro yourself", "Shanghai is"])
print(response)
```

关于 pipeline 的更多推理参数说明，请参考[这里](./docs/zh_cn/inference/pipeline.md)

# 用户教程

请阅读[快速上手](./docs/zh_cn/get_started.md)章节，了解 LMDeploy 的基本用法。

为了帮助用户更进一步了解 LMDeploy，我们准备了用户指南和进阶指南，请阅读我们的[文档](https://lmdeploy.readthedocs.io/zh-cn/latest/)：

- 用户指南
  - [推理pipeline](./docs/zh_cn/inference/pipeline.md)
  - [推理引擎 - TurboMind](./docs/zh_cn/inference/turbomind.md)
  - [推理引擎 - PyTorch](./docs/zh_cn/inference/pytorch.md)
  - [推理服务](./docs/zh_cn/serving/restful_api.md)
  - [模型量化](./docs/zh_cn/quantization)
- 进阶指南
  - 增加对话模板
  - 支持新模型
  - gemm tuning
  - [长文本推理](./docs/zh_cn/advance/long_context.md)
  - [多模型推理服务](./docs/zh_cn/serving/proxy_server.md)

## 贡献指南

我们感谢所有的贡献者为改进和提升 LMDeploy 所作出的努力。请参考[贡献指南](.github/CONTRIBUTING.md)来了解参与项目贡献的相关指引。

## 致谢

- [FasterTransformer](https://github.com/NVIDIA/FasterTransformer)
- [llm-awq](https://github.com/mit-han-lab/llm-awq)
- [vLLM](https://github.com/vllm-project/vllm)
- [DeepSpeed-MII](https://github.com/microsoft/DeepSpeed-MII)

## License

该项目采用 [Apache 2.0 开源许可证](LICENSE)。
