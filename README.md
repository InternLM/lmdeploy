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

## News 🎉

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

## Introduction

LMDeploy is a toolkit for compressing, deploying, and serving LLM, developed by the [MMRazor](https://github.com/open-mmlab/mmrazor) and [MMDeploy](https://github.com/open-mmlab/mmdeploy) teams. It has the following core features:

- **Efficient Inference Engine (TurboMind)**: Based on [FasterTransformer](https://github.com/NVIDIA/FasterTransformer), we have implemented an efficient inference engine - TurboMind, which supports the inference of LLaMA and its variant models on NVIDIA GPUs.

- **Interactive Inference Mode**: By caching the k/v of attention during multi-round dialogue processes, it remembers dialogue history, thus avoiding repetitive processing of historical sessions.

- **Multi-GPU Model Deployment and Quantization**: We provide comprehensive model deployment and quantification support, and have been validated at different scales.

- **Persistent Batch Inference**: Further optimization of model execution efficiency.

![PersistentBatchInference](https://github.com/InternLM/lmdeploy/assets/67539920/e3876167-0671-44fc-ac52-5a0f9382493e)

## Supported Models

`LMDeploy` has two inference backends, `Pytorch` and `TurboMind`.

### TurboMind

> **Note**<br />
> W4A16 inference requires Nvidia GPU with Ampere architecture or above.

|    Models    | Tensor Parallel | FP16 | KV INT8 | W4A16 | W8A8 |
| :----------: | :-------------: | :--: | :-----: | :---: | :--: |
|    Llama     |       Yes       | Yes  |   Yes   |  Yes  |  No  |
|    Llama2    |       Yes       | Yes  |   Yes   |  Yes  |  No  |
| InternLM-7B  |       Yes       | Yes  |   Yes   |  Yes  |  No  |
| InternLM-20B |       Yes       | Yes  |   Yes   |  Yes  |  No  |
|   QWen-7B    |       Yes       | Yes  |   Yes   |  No   |  No  |
|   QWen-14B   |       Yes       | Yes  |   Yes   |  No   |  No  |
| Baichuan-7B  |       Yes       | Yes  |   Yes   |  Yes  |  No  |
| Baichuan2-7B |       Yes       | Yes  |   No    |  No   |  No  |
|  Code Llama  |       Yes       | Yes  |   No    |  No   |  No  |

### Pytorch

|   Models    | Tensor Parallel | FP16 | KV INT8 | W4A16 | W8A8 |
| :---------: | :-------------: | :--: | :-----: | :---: | :--: |
|    Llama    |       Yes       | Yes  |   No    |  No   |  No  |
|   Llama2    |       Yes       | Yes  |   No    |  No   |  No  |
| InternLM-7B |       Yes       | Yes  |   No    |  No   |  No  |

## Performance

**Case I**: output token throughput with fixed input token and output token number (1, 2048)

**Case II**: request throughput with real conversation data

Test Setting: LLaMA-7B, NVIDIA A100(80G)

The output token throughput of TurboMind exceeds 2000 tokens/s, which is about 5% - 15% higher than DeepSpeed overall and outperforms huggingface transformers by up to 2.3x.
And the request throughput of TurboMind is 30% higher than vLLM.

![benchmark](https://github.com/InternLM/lmdeploy/assets/4560679/7775c518-608e-4e5b-be73-7645a444e774)

## Quick Start

### Installation

Install lmdeploy with pip ( python 3.8+) or [from source](./docs/en/build.md)

```shell
pip install lmdeploy
```

### Deploy InternLM

#### Get InternLM model

```shell
# 1. Download InternLM model

# Make sure you have git-lfs installed (https://git-lfs.com)
git lfs install
git clone https://huggingface.co/internlm/internlm-chat-7b-v1_1 /path/to/internlm-chat-7b

# if you want to clone without large files – just their pointers
# prepend your git clone with the following env var:
GIT_LFS_SKIP_SMUDGE=1

# 2. Convert InternLM model to turbomind's format, which will be in "./workspace" by default
python3 -m lmdeploy.serve.turbomind.deploy internlm-chat-7b /path/to/internlm-chat-7b

```

#### Inference by TurboMind

```shell
python -m lmdeploy.turbomind.chat ./workspace
```

> **Note**<br />
> When inferring with FP16 precision, the InternLM-7B model requires at least 15.7G of GPU memory overhead on TurboMind. <br />
> It is recommended to use NVIDIA cards such as 3090, V100, A100, etc.
> Disable GPU ECC can free up 10% memory, try `sudo nvidia-smi --ecc-config=0` and reboot system.

> **Note**<br />
> Tensor parallel is available to perform inference on multiple GPUs. Add `--tp=<num_gpu>` on `chat` to enable runtime TP.

#### Serving with gradio

```shell
python3 -m lmdeploy.serve.gradio.app ./workspace
```

![](https://github.com/InternLM/lmdeploy/assets/67539920/08d1e6f2-3767-44d5-8654-c85767cec2ab)

#### Serving with Restful API

Launch inference server by:

```shell
python3 -m lmdeploy.serve.openai.api_server ./workspace server_ip server_port --instance_num 32 --tp 1
```

Then, you can communicate with it by command line,

```shell
# restful_api_url is what printed in api_server.py, e.g. http://localhost:23333
python -m lmdeploy.serve.openai.api_client restful_api_url
```

or webui,

```shell
# restful_api_url is what printed in api_server.py, e.g. http://localhost:23333
# server_ip and server_port here are for gradio ui
# example: python -m lmdeploy.serve.gradio.app http://localhost:23333 localhost 6006 --restful_api True
python -m lmdeploy.serve.gradio.app restful_api_url server_ip --restful_api True
```

Refer to [restful_api.md](docs/en/restful_api.md) for more details.

#### Serving with Triton Inference Server

Launch inference server by:

```shell
bash workspace/service_docker_up.sh
```

Then, you can communicate with the inference server by command line,

```shell
python3 -m lmdeploy.serve.client {server_ip_addresss}:33337
```

or webui,

```shell
python3 -m lmdeploy.serve.gradio.app {server_ip_addresss}:33337
```

For the deployment of other supported models, such as LLaMA, LLaMA-2, vicuna and so on, you can find the guide from [here](docs/en/serving.md)

### Inference with PyTorch

For detailed instructions on Inference pytorch models, see [here](docs/en/pytorch.md).

#### Single GPU

```shell
python3 -m lmdeploy.pytorch.chat $NAME_OR_PATH_TO_HF_MODEL \
    --max_new_tokens 64 \
    --temperture 0.8 \
    --top_p 0.95 \
    --seed 0
```

#### Tensor Parallel with DeepSpeed

```shell
deepspeed --module --num_gpus 2 lmdeploy.pytorch.chat \
    $NAME_OR_PATH_TO_HF_MODEL \
    --max_new_tokens 64 \
    --temperture 0.8 \
    --top_p 0.95 \
    --seed 0
```

You need to install deepspeed first to use this feature.

```
pip install deepspeed
```

## Quantization

#### Weight INT4 Quantization

LMDeploy uses [AWQ](https://arxiv.org/abs/2306.00978) algorithm for model weight quantization

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
