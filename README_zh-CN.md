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

- **高效推理引擎 TurboMind**：支持 Persistent Batch(即 Continuous Batch), Blocked K/V Cache, 高效的计算 kernel，Dynamic Split&Fuse 等重要特性。

- **有状态推理**：通过缓存多轮对话过程中 attention 的 k/v，记住对话历史，从而避免重复处理历史会话。显著提升长文本多轮对话场景中的效率。

- **多 GPU 部署和量化**：我们提供了全面的模型部署和量化支持，已在不同规模上完成验证。

# 性能

## 请求处理性能(req/s)

## 静态推理性能(out tok/s)

更多设备、更多计算精度的推理 benchmark，请阅读以下链接：

- [Geforce 2080](<>)
- [Geforce RTX 3090](<>)
- [Geforce RTX 4090](<>)

# 支持的模型

`LMDeploy` 支持 2 种推理引擎： `TurboMind` 和 `PyTorch`，它们侧重不同。前者追求推理性能的极致优化，后者纯用python开发，着重降低开发者的门槛。

不同的推理引擎在支持的模型类别、计算精度方面有所差别。用户可根据实际需求选择合适的。有关两个推理引擎的架构，在[此处](<>)可以找到。

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
| ChatGLM2  |    6B     |    Yes    |   No    | Yes  |
|  Falcon   | 7B - 180B |    Yes    |   No    | Yes  |

# 快速上手

LMDeploy提供了快速安装、模型量化、离线批处理、在线推理服务等功能。每个功能只需简单的几行代码或者命令就可以完成。

<!-- toc -->

- [安装](#安装)
- [离线批处理](#离线批处理)
- [推理服务](#推理服务)
- [模型量化](#模型量化)
- [好用的工具](#好用的工具)

<!-- tocstop -->

## 安装

使用 pip ( python 3.8+) 安装 LMDeploy，或者[源码安装](./docs/zh_cn/build.md)

```shell
pip install lmdeploy
```

## 离线批处理

```shell
import lmdeploy
pipe = lmdeploy.pipeline("InternLM/internlm-chat-7b", tp=1)
response = pipe(["Hi, pls intro yourself", "Shanghai is"])
print(response)
```

支持多卡并行处理，只用修改`tp`参数。关于 pipeline 的更多推理参数说明，请参考[这里](<>)

## 推理服务

LMDeploy `api_server` 支持把模型一键封装为服务，对外提供的 RESTful API 兼容 openai 的接口。以下为服务启动和请求处理的示例：

```shell
# 启动服务
lmdeploy serve api_server internlm/internlm-chat-7b --server-port 8080 --tp 1
# 通过客户端，发送请求和接收结果
lmdeploy serve api_client http://0.0.0.0:8080
```

在上述例子中，服务启动后，在浏览器输入 `http://0.0.0.0:8080`，可在线阅读和试用 `api_server` 的各接口，也可直接查阅[文档](<>)，了解各接口的定义和使用方法。

## 模型量化

### 权重 INT4 量化

LMDeploy 使用 [AWQ](https://arxiv.org/abs/2306.00978) 算法对模型权重进行量化。

只用两行命令，就可以把一个 LLM 模型权重量化为 4bit，并在控制台与模型进行交互式对话。

```shell
lmdeploy lite auto_awq internlm/internlm-chat-7b --work-dir ./internlm-chat-7b-4bit
lmdeploy chat turbomind ./internlm-chat-7b-4bit --model-format awq --group-size 128
```

LMDeploy 4bit 量化和推理支持的显卡包括：

- 图灵架构（sm75）：20系列、T4
- 安培架构（sm80,sm86）：30系列、A10、A16、A30、A100
- Ada Lovelace架构（sm90）：40 系列

量化模型在各型号显卡上的推理速度可以从[这里](./docs/zh_cn/w4a16.md)找到。

### KV Cache INT8 量化

[点击这里](./docs/zh_cn/kv_int8.md) 查看 kv int8 使用方法、实现公式和测试结果。

### W8A8 量化

## 好用的工具

LMDeploy CLI 提供了如下便捷的工具，方便用户快速体验模型对话效果

### 控制台交互式对话

```shell
lmdeploy chat turbomind internlm/internlm-chat-7b
```

***贴一张图***

### WebUI 交互式对话

LMDeploy 使用 gradio 开发了在线对话 demo。

```shell
# 安装依赖
pip install lmdeploy[serve]
# 启动
lmdeploy serve gradio internlm/internlm-chat-7b --model-name internlm-chat-7b
```

![](https://github.com/InternLM/lmdeploy/assets/67539920/08d1e6f2-3767-44d5-8654-c85767cec2ab)

## 贡献指南

我们感谢所有的贡献者为改进和提升 LMDeploy 所作出的努力。请参考[贡献指南](.github/CONTRIBUTING.md)来了解参与项目贡献的相关指引。

## 致谢

- [FasterTransformer](https://github.com/NVIDIA/FasterTransformer)
- [llm-awq](https://github.com/mit-han-lab/llm-awq)

## License

该项目采用 [Apache 2.0 开源许可证](LICENSE)。
