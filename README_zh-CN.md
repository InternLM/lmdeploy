<div align="center">
  <img src="resources/lmdeploy-logo.svg" width="450"/>

[![docs](https://img.shields.io/badge/docs-latest-blue)](https://lmdeploy-zh-cn.readthedocs.io/zh_CN/latest/)
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

- \[2023/12\] Turbomind 支持多模态输入。[Gradio Demo](./examples/vl/README.md)
- \[2023/11\] Turbomind 支持直接读取 Huggingface 模型。点击[这里](./docs/en/load_hf.md)查看使用方法
- \[2023/11\] TurboMind 重磅升级。包括：Paged Attention、更快的且不受序列最大长度限制的 attention kernel、2+倍快的 KV8 kernels、Split-K decoding (Flash Decoding) 和 支持 sm_75 架构的 W4A16
- \[2023/09\] TurboMind 支持 Qwen-14B
- \[2023/09\] TurboMind 支持 InternLM-20B 模型
- \[2023/09\] TurboMind 支持 Code Llama 所有功能：代码续写、填空、对话、Python专项。点击[这里](./docs/zh_cn/supported_models/codellama.md)阅读部署方法
- \[2023/09\] TurboMind 支持 Baichuan2-7B
- \[2023/08\] TurboMind 支持 flash-attention2
- \[2023/08\] TurboMind 支持 Qwen-7B，动态NTK-RoPE缩放，动态logN缩放
- \[2023/08\] TurboMind 支持 Windows (tp=1)
- \[2023/08\] TurboMind 支持 4-bit 推理，速度是 FP16 的 2.4 倍，是目前最快的开源实现🚀。部署方式请看[这里](./docs/zh_cn/w4a16.md)
- \[2023/08\] LMDeploy 开通了 [HuggingFace Hub](https://huggingface.co/lmdeploy) ，提供开箱即用的 4-bit 模型
- \[2023/08\] LMDeploy 支持使用 [AWQ](https://arxiv.org/abs/2306.00978) 算法进行 4-bit 量化
- \[2023/07\] TurboMind 支持使用 GQA 的 Llama-2 70B 模型
- \[2023/07\] TurboMind 支持 Llama-2 7B/13B 模型
- \[2023/07\] TurboMind 支持 InternLM 的 Tensor Parallel 推理

______________________________________________________________________

# 简介

LMDeploy 由 [MMDeploy](https://github.com/open-mmlab/mmdeploy) 和 [MMRazor](https://github.com/open-mmlab/mmrazor) 团队联合开发，是涵盖了 LLM 任务的全套轻量化、部署和服务解决方案。
这个强大的工具箱提供以下核心功能：

- **高效推理引擎 TurboMind**：开发了 Persistent Batch(即 Continuous Batch)，Blocked K/V Cache，动态拆分和融合，张量并行，高效的计算 kernel等重要特性，保障了 LLMs 推理时的高吞吐和低延时。

- **有状态推理**：通过缓存多轮对话过程中 attention 的 k/v，记住对话历史，从而避免重复处理历史会话。显著提升长文本多轮对话场景中的效率。

- **量化**：LMDeploy 支持多种量化方式和高效的量化模型推理。在不同规模的模型上，验证了量化的可靠性。

# 性能

LMDeploy TurboMind 引擎拥有卓越的推理能力，在各种规模的模型上，每秒处理的请求数是 vLLM 的 1.36 ~ 1.85 倍。在静态推理能力方面，TurboMind 4bit 模型推理速度（out token/s）远高于 FP16/BF16 推理。在小 batch 时，提高到 2.4 倍。

![v0 1 0-benchmark](https://github.com/InternLM/lmdeploy/assets/4560679/f4d218f9-db3b-4ceb-ab50-97cb005b3ac9)

更多设备、更多计算精度、更多setting下的的推理 benchmark，请参考以下链接：

- [A100](./docs/en/benchmark/a100_fp16.md)

# 支持的模型

`LMDeploy` 支持 2 种推理引擎： `TurboMind` 和 `PyTorch`，它们侧重不同。前者追求推理性能的极致优化，后者纯用python开发，着重降低开发者的门槛。

不同的推理引擎在支持的模型类别、计算精度方面有所差别。用户可根据实际需求选择合适的。

## TurboMind 支持的模型

|        模型        | 模型规模 | FP16/BF16 | KV INT8 | W4A16 |
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

### PyTorch 支持的模型

|   模型    | 模型规模  | FP16/BF16 | KV INT8 | W8A8 |
| :-------: | :-------: | :-------: | :-----: | :--: |
|   Llama   | 7B - 65B  |    Yes    |   No    | Yes  |
|  Llama2   | 7B - 70B  |    Yes    |   No    | Yes  |
| InternLM  | 7B - 20B  |    Yes    |   No    | Yes  |
| Baichuan2 | 7B - 13B  |    Yes    |   No    | Yes  |
| ChatGLM2  |    6B     |    Yes    |   No    |  No  |
|  Falcon   | 7B - 180B |    Yes    |   No    |  No  |

# 用户教程

请阅读[快速上手](<>)章节，了解 LMDeploy 的基本用法。

为了帮助用户更进一步了解 LMDeploy，我们准备了用户指南和进阶指南，请阅读我们的[文档](<>)：

- 用户指南
  - 推理pipeline
  - [推理引擎 - TurboMind](<>)
  - 推理引擎 - PyTorch
  - [推理服务](<>)
  - [模型量化](<>)
- 进阶指南
  - 增加对话模板
  - 支持新模型
  - gemm tuning
  - 长文本推理

## 贡献指南

我们感谢所有的贡献者为改进和提升 LMDeploy 所作出的努力。请参考[贡献指南](.github/CONTRIBUTING.md)来了解参与项目贡献的相关指引。

## 致谢

- [FasterTransformer](https://github.com/NVIDIA/FasterTransformer)
- [llm-awq](https://github.com/mit-han-lab/llm-awq)
- [vLLM](https://github.com/vllm-project/vllm)
- [DeepSpeed-MII](https://github.com/microsoft/DeepSpeed-MII)

## License

该项目采用 [Apache 2.0 开源许可证](LICENSE)。
