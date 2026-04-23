<div align="center">
  <img src="docs/en/_static/image/lmdeploy-logo.svg" width="450"/>

[![PyPI](https://img.shields.io/pypi/v/lmdeploy)](https://pypi.org/project/lmdeploy)
![PyPI - Downloads](https://img.shields.io/pypi/dm/lmdeploy)
[![license](https://img.shields.io/github/license/InternLM/lmdeploy.svg)](https://github.com/InternLM/lmdeploy/tree/main/LICENSE)
[![issue resolution](https://img.shields.io/github/issues-closed-raw/InternLM/lmdeploy)](https://github.com/InternLM/lmdeploy/issues)
[![open issues](https://img.shields.io/github/issues-raw/InternLM/lmdeploy)](https://github.com/InternLM/lmdeploy/issues)

[📘Documentation](https://lmdeploy.readthedocs.io/zh-cn/latest/) |
[🛠️Quick Start](https://lmdeploy.readthedocs.io/zh-cn/latest/get_started/get_started.html) |
[🤔Reporting Issues](https://github.com/InternLM/lmdeploy/issues/new/choose)

[English](README.md) | 简体中文 | [日本語](README_ja.md)

👋 join us on [![Static Badge](https://img.shields.io/badge/-grey?style=social&logo=wechat&label=WeChat)](https://cdn.vansin.top/internlm/lmdeploy.jpg)
[![Static Badge](https://img.shields.io/badge/-grey?style=social&logo=twitter&label=Twitter)](https://twitter.com/intern_lm)
[![Static Badge](https://img.shields.io/badge/-grey?style=social&logo=discord&label=Discord)](https://discord.gg/xa29JuW87d)

</div>

______________________________________________________________________

## 最新进展 🎉

<details open>
<summary><b>2026</b></summary>

- \[2026/04\] 由于 LMDeploy 在 PyPI 上的项目存储配额已满，新版本目前无法上传预编译安装包（wheels）。用户可以通过 [GitHub Releases](https://github.com/InternLM/lmdeploy/releases) 页面下载安装包，或者通过源码安装等方式使用最新版本；预编译包恢复上传后我们会另行通知。受影响版本：>=0.12.2
- \[2026/02\] 支持 [Qwen3.5](https://huggingface.co/collections/Qwen/qwen35)
- \[2026/02\] 支持 [vllm-project/llm-compressor](https://github.com/vllm-project/llm-compressor) 4bit 对称和非对称量化。 具体操作指南详见[此处](./docs/zh_cn/quantization/llm_compressor.md)

</details>

<details close>
<summary><b>2025</b></summary>

- 【2025年9月】TurboMind 引擎支持 MXFP4，适用于 NVIDIA V100 及以上 GPU。在 H800 上推理 openai gpt-oss 模型，性能可达 vLLM 的 1.5倍！
- 【2025年6月】深度优化 FP8 MoE 模型推理
- 【2025年6月】集成[DLSlime](https://github.com/DeepLink-org/DLSlime)和[Mooncake](https://github.com/kvcache-ai/Mooncake)，实现DeepSeek PD分离部署，向两个团队表示诚挚的感谢！
- 【2025年4月】集成deepseek-ai组件FlashMLA、DeepGemm、DeepEP、MicroBatch、eplb等，提升DeepSeek推理性能
- 【2025年1月】新增对DeepSeek V3及R1的支持

</details>

<details close>
<summary><b>2024</b></summary>

- \[2024/11\] PyTorch engine 支持 Mono-InternVL 模型
- \[2024/10\] PyTorchEngine 在 ascend 平台上支持了图模式，推理性能提高了 1 倍
- \[2024/09\] LMDeploy PyTorchEngine 增加了对 [华为 Ascend](docs/zh_cn/get_started/ascend/get_started.md) 的支持。支持的模型请见[这里](docs/zh_cn/supported_models/supported_models.md)
- \[2024/09\] 通过引入 CUDA Graph，LMDeploy PyTorchEngine 在 Llama3-8B 推理上实现了 1.3 倍的加速
- \[2024/08\] LMDeploy现已集成至 [modelscope/swift](https://github.com/modelscope/swift)，成为 VLMs 推理的默认加速引擎
- \[2024/07\] 支持 Llama3.1 8B 和 70B 模型，以及工具调用功能
- \[2024/07\] 支持 [InternVL2](docs/zh_cn/multi_modal/internvl.md) 全系列模型，[InternLM-XComposer2.5](docs/zh_cn/multi_modal/xcomposer2d5.md) 模型和 InternLM2.5 的 [function call 功能](docs/zh_cn/llm/api_server_tools.md)
- \[2024/06\] PyTorch engine 支持了 DeepSeek-V2 和若干 VLM 模型推理, 比如 CogVLM2，Mini-InternVL，LlaVA-Next
- \[2024/05\] 在多 GPU 上部署 VLM 模型时，支持把视觉部分的模型均分到多卡上
- \[2024/05\] 支持InternVL v1.5, LLaVa, InternLMXComposer2 等 VLMs 模型的 4bit 权重量化和推理
- \[2024/04\] 支持 Llama3 和 InternVL v1.1, v1.2，MiniGemini，InternLM-XComposer2 等 VLM 模型
- \[2024/04\] TurboMind 支持 kv cache int4/int8 在线量化和推理，适用已支持的所有型号显卡。详情请参考[这里](docs/zh_cn/quantization/kv_quant.md)
- \[2024/04\] TurboMind 引擎升级，优化 GQA 推理。[internlm2-20b](https://huggingface.co/internlm/internlm2-20b) 推理速度达 16+ RPS，约是 vLLM 的 1.8 倍
- \[2024/04\] 支持 Qwen1.5-MOE 和 dbrx.
- \[2024/03\] 支持 DeepSeek-VL 的离线推理 pipeline 和推理服务
- \[2024/03\] 支持视觉-语言模型（VLM）的离线推理 pipeline 和推理服务
- \[2024/02\] 支持 Qwen 1.5、Gemma、Mistral、Mixtral、Deepseek-MOE 等模型
- \[2024/01\] [OpenAOE](https://github.com/InternLM/OpenAOE) 发布，支持无缝接入[LMDeploy Serving Service](docs/zh_cn/llm/api_server.md)
- \[2024/01\] 支持多模型、多机、多卡推理服务。使用方法请参考[此处](docs/zh_cn/llm/proxy_server.md)
- \[2024/01\] 增加 [PyTorch 推理引擎](./docs/zh_cn/inference/pytorch.md)，作为 TurboMind 引擎的补充。帮助降低开发门槛，和快速实验新特性、新技术

</details>

<details close>
<summary><b>2023</b></summary>

- \[2023/12\] Turbomind 支持多模态输入
- \[2023/11\] Turbomind 支持直接读取 Huggingface 模型。点击[这里](docs/zh_cn/inference/load_hf.md)查看使用方法
- \[2023/11\] TurboMind 重磅升级。包括：Paged Attention、更快的且不受序列最大长度限制的 attention kernel、2+倍快的 KV8 kernels、Split-K decoding (Flash Decoding) 和 支持 sm_75 架构的 W4A16
- \[2023/09\] TurboMind 支持 Qwen-14B
- \[2023/09\] TurboMind 支持 InternLM-20B 模型
- \[2023/09\] TurboMind 支持 Code Llama 所有功能：代码续写、填空、对话、Python专项。点击[这里](./docs/zh_cn/llm/codellama.md)阅读部署方法
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

- **卓越的兼容性**: LMDeploy 支持 [KV Cache 量化](docs/zh_cn/quantization/kv_quant.md), [AWQ](docs/zh_cn/quantization/w4a16.md) 和 [Automatic Prefix Caching](docs/zh_cn/inference/turbomind_config.md) 同时使用。

# 性能

LMDeploy TurboMind 引擎拥有卓越的推理能力，在各种规模的模型上，每秒处理的请求数是 vLLM 的 1.36 ~ 1.85 倍。在静态推理能力方面，TurboMind 4bit 模型推理速度（out token/s）远高于 FP16/BF16 推理。在小 batch 时，提高到 2.4 倍。

![v0 1 0-benchmark](https://github.com/InternLM/lmdeploy/assets/4560679/8e455cf1-a792-4fa8-91a2-75df96a2a5ba)

# 支持的模型

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
  <li>DeepSeek-V3 (685B)</li>
  <li>DeepSeek-V3.2 (685B)</li>
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
  <li>GLM-4.7-Flash (30B)</li>
  <li>GLM-5 (754B)</li>
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
  <li>Qwen3.5 (0.8B - 397B)</li>
  <li>DeepSeek-VL (7B)</li>
  <li>DeepSeek-VL2 (3B, 16B, 27B)</li>
  <li>InternVL-Chat (v1.1-v1.5)</li>
  <li>InternVL2 (1B-76B)</li>
  <li>InternVL2.5(MPO) (1B-78B)</li>
  <li>InternVL3 (1B-78B)</li>
  <li>InternVL3.5 (1B-241BA28B)</li>
  <li>Intern-S1 (241B)</li>
  <li>Intern-S1-mini (8.3B)</li>
  <li>Intern-S1-Pro (1TB)</li>
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

LMDeploy 支持 2 种推理引擎： [TurboMind](./docs/zh_cn/inference/turbomind.md) 和 [PyTorch](./docs/zh_cn/inference/pytorch.md)，它们侧重不同。前者追求推理性能的极致优化，后者纯用python开发，着重降低开发者的门槛。

它们在支持的模型类别、计算精度方面有所差别。用户可参考[这里](./docs/zh_cn/supported_models/supported_models.md), 查阅每个推理引擎的能力，并根据实际需求选择合适的。

# 快速开始 [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1Dh-YlSwg78ZO3AlleO441NF_QP2shs95#scrollTo=YALmXnwCG1pQ)

## 安装

我们推荐在一个干净的conda环境下（python3.10 - 3.13），安装 lmdeploy：

```shell
conda create -n lmdeploy python=3.12 -y
conda activate lmdeploy
pip install lmdeploy
```

自 **v0.13.0** 起，**PyPI** 上默认预编译 wheel 基于 **CUDA 12.8** 构建，一般用户（含 GeForce RTX 50 系列）使用上方的 `pip install lmdeploy` 即可。

## 离线批处理

```python
import lmdeploy
with lmdeploy.pipeline("internlm/internlm3-8b-instruct") as pipe:
    response = pipe(["Hi, pls intro yourself", "Shanghai is"])
    print(response)
```

> \[!NOTE\]
> LMDeploy 默认从 HuggingFace 上面下载模型，如果要从 ModelScope 上面下载模型，请通过命令 `pip install modelscope` 安装ModelScope，并设置环境变量：
>
> `export LMDEPLOY_USE_MODELSCOPE=True`
>
> 如果要从 openMind Hub 上面下载模型，请通过命令 `pip install openmind_hub` 安装openMind Hub，并设置环境变量：
>
> `export LMDEPLOY_USE_OPENMIND_HUB=True`

关于 pipeline 的更多推理参数说明，请参考[这里](docs/zh_cn/llm/pipeline.md)

# 用户教程

请阅读[快速上手](docs/zh_cn/get_started/get_started.md)章节，了解 LMDeploy 的基本用法。

为了帮助用户更进一步了解 LMDeploy，我们准备了用户指南和进阶指南，请阅读我们的[文档](https://lmdeploy.readthedocs.io/zh-cn/latest/)：

- 用户指南
  - [LLM 推理 pipeline](docs/zh_cn/llm/pipeline.md) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1Dh-YlSwg78ZO3AlleO441NF_QP2shs95#scrollTo=YALmXnwCG1pQ)
  - [VLM 推理 pipeline](docs/zh_cn/multi_modal/vl_pipeline.md) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1nKLfnPeDA3p-FMNw2NhI-KOpk7-nlNjF?usp=sharing)
  - [LLM 推理服务](docs/zh_cn/llm/api_server.md)
  - [VLM 推理服务](docs/zh_cn/multi_modal/api_server_vl.md)
  - [模型量化](./docs/zh_cn/quantization)
- 进阶指南
  - [推理引擎 - TurboMind](./docs/zh_cn/inference/turbomind.md)
  - [推理引擎 - PyTorch](./docs/zh_cn/inference/pytorch.md)
  - [自定义对话模板](./docs/zh_cn/advance/chat_template.md)
  - [支持新模型](./docs/zh_cn/advance/pytorch_new_model.md)
  - gemm tuning
  - [长文本推理](./docs/zh_cn/advance/long_context.md)
  - [多模型推理服务](docs/zh_cn/llm/proxy_server.md)

# 社区项目

- 使用LMDeploy在英伟达Jetson系列板卡部署大模型：[LMDeploy-Jetson](https://github.com/BestAnHongjun/LMDeploy-Jetson)
- 使用 LMDeploy 和 BentoML 部署大模型的示例项目：[BentoLMDeploy](https://github.com/bentoml/BentoLMDeploy)

# 贡献指南

我们感谢所有的贡献者为改进和提升 LMDeploy 所作出的努力。请参考[贡献指南](.github/CONTRIBUTING.md)来了解参与项目贡献的相关指引。

# 致谢

- [FasterTransformer](https://github.com/NVIDIA/FasterTransformer)
- [llm-awq](https://github.com/mit-han-lab/llm-awq)
- [vLLM](https://github.com/vllm-project/vllm)
- [DeepSpeed-MII](https://github.com/microsoft/DeepSpeed-MII)

# 引用

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

# 开源许可证

该项目采用 [Apache 2.0 开源许可证](LICENSE)。
