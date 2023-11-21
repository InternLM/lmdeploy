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

## 更新 🎉

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

## 简介

LMDeploy 由 [MMDeploy](https://github.com/open-mmlab/mmdeploy) 和 [MMRazor](https://github.com/open-mmlab/mmrazor) 团队联合开发，是涵盖了 LLM 任务的全套轻量化、部署和服务解决方案。
这个强大的工具箱提供以下核心功能：

- **高效推理引擎 TurboMind**：基于 [FasterTransformer](https://github.com/NVIDIA/FasterTransformer)，我们实现了高效推理引擎 TurboMind，支持 InternLM、LLaMA、vicuna等模型在 NVIDIA GPU 上的推理。

- **交互推理方式**：通过缓存多轮对话过程中 attention 的 k/v，记住对话历史，从而避免重复处理历史会话。

- **多 GPU 部署和量化**：我们提供了全面的模型部署和量化支持，已在不同规模上完成验证。

- **persistent batch 推理**：进一步优化模型执行效率。

  ![PersistentBatchInference](https://github.com/InternLM/lmdeploy/assets/67539920/e3876167-0671-44fc-ac52-5a0f9382493e)

## 支持的模型

`LMDeploy` 支持 `TurboMind` 和 `Pytorch` 两种推理后端。运行`lmdeploy list`可查看支持模型列表

### TurboMind

> **Note**<br />
> W4A16 推理需要 Ampere 及以上架构的 Nvidia GPU

|     模型     | 模型并行 | FP16 | KV INT8 | W4A16 | W8A8 |
| :----------: | :------: | :--: | :-----: | :---: | :--: |
|    Llama     |   Yes    | Yes  |   Yes   |  Yes  |  No  |
|    Llama2    |   Yes    | Yes  |   Yes   |  Yes  |  No  |
|    SOLAR     |   Yes    | Yes  |   Yes   |  Yes  |  No  |
| InternLM-7B  |   Yes    | Yes  |   Yes   |  Yes  |  No  |
| InternLM-20B |   Yes    | Yes  |   Yes   |  Yes  |  No  |
|   QWen-7B    |   Yes    | Yes  |   Yes   |  Yes  |  No  |
|   QWen-14B   |   Yes    | Yes  |   Yes   |  Yes  |  No  |
| Baichuan-7B  |   Yes    | Yes  |   Yes   |  Yes  |  No  |
| Baichuan2-7B |   Yes    | Yes  |   Yes   |  Yes  |  No  |
|  Code Llama  |   Yes    | Yes  |   No    |  No   |  No  |

### Pytorch

|    模型     | 模型并行 | FP16 | KV INT8 | W4A16 | W8A8 |
| :---------: | :------: | :--: | :-----: | :---: | :--: |
|    Llama    |   Yes    | Yes  |   No    |  No   |  No  |
|   Llama2    |   Yes    | Yes  |   No    |  No   |  No  |
| InternLM-7B |   Yes    | Yes  |   No    |  No   |  No  |

## 性能

**场景一**: 固定的输入、输出token数（1,2048），测试 output token throughput

**场景二**: 使用真实数据，测试 request throughput

测试配置：LLaMA-7B, NVIDIA A100(80G)

TurboMind 的 output token throughput 超过 2000 token/s, 整体比 DeepSpeed 提升约 5% - 15%，比 huggingface transformers 提升 2.3 倍
在 request throughput 指标上，TurboMind 的效率比 vLLM 高 30%

![benchmark](https://github.com/InternLM/lmdeploy/assets/4560679/7775c518-608e-4e5b-be73-7645a444e774)

## 快速上手

### 安装

使用 pip ( python 3.8+) 安装 LMDeploy，或者[源码安装](./docs/zh_cn/build.md)

```shell
pip install lmdeploy
```

> **Note**<br />
> `pip install lmdeploy`默认安装runtime依赖包，使用lmdeploy的lite和serve功能时，用户需要安装额外依赖包。例如: `pip install lmdeploy[lite]` 会额外安装`lmdeploy.lite`模块的依赖包
>
> - `all`: 安装`lmdeploy`所有依赖包，具体可查看`requirements.txt`
> - `lite`: 额外安装`lmdeploy.lite`模块的依赖包，具体可查看`requirements/lite.txt`
> - `serve`: 额外安装`lmdeploy.serve`模块的依赖包，具体可查看`requirements/serve.txt`

### 部署 InternLM

使用 TurboMind 推理模型需要先将模型转化为 TurboMind 的格式，目前支持在线转换和离线转换两种形式。在线转换可以直接加载 Huggingface 模型，离线转换需需要先保存模型再加载。

下面以 [internlm/internlm-chat-7b-v1_1](https://huggingface.co/internlm/internlm-chat-7b-v1_1) 为例，展示在线转换的使用方式。其他方式可参考[load_hf.md](docs/zh_cn/load_hf.md)

#### 使用 turbomind 推理

```shell
lmdeploy chat turbomind internlm/internlm-chat-7b-v1_1 --model-name internlm-chat-7b
```

> **Note**<br /> internlm/internlm-chat-7b-v1_1 会自动下载到 `.cache` 文件夹，这里也可以传下载好的路径。

> **Note**<br />
> turbomind 在使用 FP16 精度推理 InternLM-7B 模型时，显存开销至少需要 15.7G。建议使用 3090, V100，A100等型号的显卡。<br />
> 关闭显卡的 ECC 可以腾出 10% 显存，执行 `sudo nvidia-smi --ecc-config=0` 重启系统生效。

> **Note**<br />
> 使用 Tensor 并发可以利用多张 GPU 进行推理。在 `chat` 时添加参数 `--tp=<num_gpu>` 可以启动运行时 TP。

#### 启动 gradio server

```shell
# 安装lmdeploy额外依赖
pip install lmdeploy[serve]

lmdeploy serve gradio internlm/internlm-chat-7b-v1_1 --model-name internlm-chat-7b
```

![](https://github.com/InternLM/lmdeploy/assets/67539920/08d1e6f2-3767-44d5-8654-c85767cec2ab)

#### 通过 Restful API 部署服务

使用下面的命令启动推理服务：

```shell
# 安装lmdeploy额外依赖
pip install lmdeploy[serve]

lmdeploy serve api_server internlm/internlm-chat-7b-v1_1 --model-name internlm-chat-7b --instance_num 32 --tp 1
```

你可以通过命令行方式与推理服务进行对话：

```shell
# api_server_url is what printed in api_server.py, e.g. http://localhost:23333
lmdeploy serve api_client api_server_url
```

也可以通过 WebUI 方式来对话：

```shell
# api_server_url is what printed in api_server.py, e.g. http://localhost:23333
# server_ip and server_port here are for gradio ui
# example: lmdeploy serve gradio http://localhost:23333 --server_name localhost --server_port 6006
lmdeploy serve gradio api_server_url --server_name ${gradio_ui_ip} --server_port ${gradio_ui_port}
```

更多详情可以查阅 [restful_api.md](docs/zh_cn/restful_api.md)。

### 基于 PyTorch 的推理

你必须确保环境中有安装 deepspeed：

```
pip install deepspeed
```

#### 单个 GPU

```shell
lmdeploy chat torch $NAME_OR_PATH_TO_HF_MODEL\
    --max_new_tokens 64 \
    --temperture 0.8 \
    --top_p 0.95 \
    --seed 0
```

#### 使用 DeepSpeed 实现张量并行

```shell
deepspeed --module --num_gpus 2 lmdeploy.pytorch.chat \
    $NAME_OR_PATH_TO_HF_MODEL \
    --max_new_tokens 64 \
    --temperture 0.8 \
    --top_p 0.95 \
    --seed 0
```

## 量化部署

#### 权重 INT4 量化

LMDeploy 使用 [AWQ](https://arxiv.org/abs/2306.00978) 算法对模型权重进行量化

[点击这里](./docs/zh_cn/w4a16.md) 查看 weight int4 用法测试结果。

#### KV Cache INT8 量化

[点击这里](./docs/zh_cn/kv_int8.md) 查看 kv int8 使用方法、实现公式和测试结果。

> **Warning**<br />
> 量化部署不支持运行时 Tensor 并发。如果希望使用 Tensor 并发，需要在 deploy 时配置 tp 参数。

## 贡献指南

我们感谢所有的贡献者为改进和提升 LMDeploy 所作出的努力。请参考[贡献指南](.github/CONTRIBUTING.md)来了解参与项目贡献的相关指引。

## 致谢

- [FasterTransformer](https://github.com/NVIDIA/FasterTransformer)
- [llm-awq](https://github.com/mit-han-lab/llm-awq)

## License

该项目采用 [Apache 2.0 开源许可证](LICENSE)。
