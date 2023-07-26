<div align="center">
  <img src="resources/lmdeploy-logo.png" width="450"/>

English | [简体中文](README_zh-CN.md)

</div>

<p align="center">
    👋 join us on <a href="https://discord.gg/xa29JuW87d" target="_blank">Discord</a> and <a href="https://r.vansin.top/?r=internwx" target="_blank">WeChat</a>
</p>

______________________________________________________________________

## News 🎉

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

## Performance

**Case I**: output token throughput with fixed input token and output token number (1, 2048)

**Case II**: request throughput with real conversation data

Test Setting: LLaMA-7B, NVIDIA A100(80G)

The output token throughput of TurboMind exceeds 2000 tokens/s, which is about 5% - 15% higher than DeepSpeed overall and outperforms huggingface transformers by up to 2.3x.
And the request throughput of TurboMind is 30% higher than vLLM.

![benchmark](https://github.com/InternLM/lmdeploy/assets/4560679/7775c518-608e-4e5b-be73-7645a444e774)

## Quick Start

### Installation

Below are quick steps for installation:

```shell
conda create -n lmdeploy python=3.10 -y
conda activate lmdeploy
git clone https://github.com/InternLM/lmdeploy.git
cd lmdeploy
pip install -e .
```

### Deploy InternLM

#### Get InternLM model

```shell
# 1. Download InternLM model

# Make sure you have git-lfs installed (https://git-lfs.com)
git lfs install
git clone https://huggingface.co/internlm/internlm-chat-7b /path/to/internlm-chat-7b

# if you want to clone without large files – just their pointers
# prepend your git clone with the following env var:
GIT_LFS_SKIP_SMUDGE=1

# 2. Convert InternLM model to turbomind's format, which will be in "./workspace" by default
python3 -m lmdeploy.serve.turbomind.deploy internlm-chat-7b /path/to/internlm-chat-7b

```

#### Inference by TurboMind

```shell
docker run --gpus all --rm -v $(pwd)/workspace:/workspace -it openmmlab/lmdeploy:latest \
    python3 -m lmdeploy.turbomind.chat /workspace
```

> **Note**
> When inferring with FP16 precision, the InternLM-7B model requires at least 15.7G of GPU memory overhead on TurboMind. It is recommended to use NVIDIA cards such as 3090, V100, A100, etc.

> **Note**
> Tensor parallel can be used to perform inference on multiple GPUs. Add `--tp=<num_gpu>` on `deploy` to enable static TP or add `--tp=<num_gpu>` on `chat` to enable runtime TP.

#### Serving

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
python3 -m lmdeploy.app {server_ip_addresss}:33337 internlm
```

![](https://github.com/InternLM/lmdeploy/assets/67539920/08d1e6f2-3767-44d5-8654-c85767cec2ab)

For the deployment of other supported models, such as LLaMA, LLaMA-2, vicuna and so on, you can find the guide from [here](docs/en/serving.md)

### Inference with PyTorch

You have to install deepspeed first before running with PyTorch.

```
pip install deepspeed
```

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

## Quantization

In fp16 mode, kv_cache int8 quantization can be enabled, and a single card can serve more users.
First execute the quantization script, and the quantization parameters are stored in the `workspace/triton_models/weights` transformed by `deploy.py`.

```
python3 -m lmdeploy.lite.apis.kv_qparams \
  --model $HF_MODEL \
  --output_dir $DEPLOY_WEIGHT_DIR \
  --symmetry True \   # Whether to use symmetric or asymmetric quantization.
  --offload  False \  # Whether to offload some modules to CPU to save GPU memory.
  --num_tp 1 \   # The number of GPUs used for tensor parallelism
```

Then adjust `workspace/triton_models/weights/config.ini`

- `use_context_fmha` changed to 0, means off
- `quant_policy` is set to 4. This parameter defaults to 0, which means it is not enabled

Here is [quantization test results](./docs/zh_cn/quantization.md).

> **Warning**
> runtime Tesnor Parallel for quantilized model is not available. Please setup `--tp` on `deploy` to enable static TP.

## Contributing

We appreciate all contributions to LMDeploy. Please refer to [CONTRIBUTING.md](.github/CONTRIBUTING.md) for the contributing guideline.

## Acknowledgement

- [FasterTransformer](https://github.com/NVIDIA/FasterTransformer)

## License

This project is released under the [Apache 2.0 license](LICENSE).
