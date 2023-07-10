<div align="center">
  <img src="resources/lmdeploy-logo.png" width="450"/>

English | [简体中文](README_zh-CN.md)

</div>

<div align="center">
  <a href="https://openmmlab.medium.com/" style="text-decoration:none;">
    <img src="https://user-images.githubusercontent.com/25839884/219255827-67c1a27f-f8c5-46a9-811d-5e57448c61d1.png" width="3%" alt="" /></a>
  <img src="https://user-images.githubusercontent.com/25839884/218346358-56cc8e2f-a2b8-487f-9088-32480cceabcf.png" width="3%" alt="" />
  <a href="https://discord.com/channels/1037617289144569886/1046608014234370059" style="text-decoration:none;">
    <img src="https://user-images.githubusercontent.com/25839884/218347213-c080267f-cbb6-443e-8532-8e1ed9a58ea9.png" width="3%" alt="" /></a>
  <img src="https://user-images.githubusercontent.com/25839884/218346358-56cc8e2f-a2b8-487f-9088-32480cceabcf.png" width="3%" alt="" />
  <a href="https://twitter.com/OpenMMLab" style="text-decoration:none;">
    <img src="https://user-images.githubusercontent.com/25839884/218346637-d30c8a0f-3eba-4699-8131-512fb06d46db.png" width="3%" alt="" /></a>
  <img src="https://user-images.githubusercontent.com/25839884/218346358-56cc8e2f-a2b8-487f-9088-32480cceabcf.png" width="3%" alt="" />
  <a href="https://www.youtube.com/openmmlab" style="text-decoration:none;">
    <img src="https://user-images.githubusercontent.com/25839884/218346691-ceb2116a-465a-40af-8424-9f30d2348ca9.png" width="3%" alt="" /></a>
  <img src="https://user-images.githubusercontent.com/25839884/218346358-56cc8e2f-a2b8-487f-9088-32480cceabcf.png" width="3%" alt="" />
  <a href="https://space.bilibili.com/1293512903" style="text-decoration:none;">
    <img src="https://user-images.githubusercontent.com/25839884/219026751-d7d14cce-a7c9-4e82-9942-8375fca65b99.png" width="3%" alt="" /></a>
  <img src="https://user-images.githubusercontent.com/25839884/218346358-56cc8e2f-a2b8-487f-9088-32480cceabcf.png" width="3%" alt="" />
  <a href="https://www.zhihu.com/people/openmmlab" style="text-decoration:none;">
    <img src="https://user-images.githubusercontent.com/25839884/219026120-ba71e48b-6e94-4bd4-b4e9-b7d175b5e362.png" width="3%" alt="" /></a>
</div>

## Introduction

LMDeploy is a toolkit for compressing, deploying, and serving LLM, developed by the [MMRazor](https://github.com/open-mmlab/mmrazor) and [MMDeploy](https://github.com/open-mmlab/mmdeploy) teams. It has the following core features:

- **Efficient Inference Engine (TurboMind)**: Based on [FasterTransformer](https://github.com/NVIDIA/FasterTransformer), we have implemented an efficient inference engine - TurboMind, which supports the inference of LLaMA and its variant models on NVIDIA GPUs.

- **Interactive Inference Mode**: By caching the k/v of attention during multi-round dialogue processes, it remembers dialogue history, thus avoiding repetitive processing of historical sessions.

- **Multi-GPU Model Deployment and Quantization**: We provide comprehensive model deployment and quantification support, and have been validated at different scales.

- **Persistent Batch Inference**: Further optimization of model execution efficiency.

![PersistentBatchInference](https://github.com/InternLM/lmdeploy/assets/67539920/e3876167-0671-44fc-ac52-5a0f9382493e)

## Performance

As shown in the figure below, we have compared the token generation speed among facebookresearch/llama, HuggingFace Transformers, and DeepSpeed on the 7B model.

Target Device: NVIDIA A100(80G)

Metrics: Throughput (token/s)

Test Data: The number of input tokens is 1, and the number of generated tokens is 2048

The throughput of TurboMind exceeds 2000 tokens/s, which is about 5% - 15% higher than DeepSpeed overall and outperforms huggingface transformers by up to 2.3x

![benchmark](https://user-images.githubusercontent.com/12756472/251422522-e94a3db9-eb16-432a-8d8c-078945e7b99a.png)

## Quick Start

### Installation

Below are quick steps for installation:

```shell
conda create -n lmdeploy python=3.10
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
git clone https://huggingface.co/internlm/internlm-7b /path/to/internlm-7b

# if you want to clone without large files – just their pointers
# prepend your git clone with the following env var:
GIT_LFS_SKIP_SMUDGE=1

# 2. Convert InternLM model to turbomind's format, which will be in "./workspace" by default
python3 -m lmdeploy.serve.turbomind.deploy internlm-7b /path/to/internlm-7b hf

```

#### Inference by TurboMind

```shell
docker run --gpus all --rm -v $(pwd)/workspace:/workspace -it openmmlab/lmdeploy:latest \
    python3 -m lmdeploy.turbomind.chat internlm /workspace
```

```{note}
When inferring with FP16 precision, the InternLM-7B model requires at least 22.7G of GPU memory overhead on TurboMind. It is recommended to use NVIDIA cards such as 3090, V100, A100, etc.
```

#### Serving

Launch inference server by:

```shell
bash workspace/service_docker_up.sh
```

Then, you can communicate with the inference server by command line,

```shell
python3 lmdeploy.serve.client {server_ip_addresss}:33337 internlm
```

or webui,

```
python3 lmdeploy.app {server_ip_addresss}:33337 internlm
```

![](https://github.com/InternLM/lmdeploy/assets/67539920/08d1e6f2-3767-44d5-8654-c85767cec2ab)

For the deployment of other supported models, such as LLaMA, vicuna and Baichuan, you can find the guide from [here](docs/en/serving.md)

### Inference with PyTorch

#### Single GPU

```shell
python3 -m lmdeploy.torch.chat $NAME_OR_PATH_TO_HF_MODEL\
    --max_new_tokens 64 \
    --temperture 0.8 \
    --top_p 0.95 \
    --seed 0
```

#### Tensor Parallel with DeepSpeed

```shell
deepspeed --module --num_gpus 2 lmdeploy.torch.chat \
    $NAME_OR_PATH_TO_HF_MODEL \
    --max_new_tokens 64 \
    --temperture 0.8 \
    --top_p 0.95 \
    --seed 0
```

## Quantization

In fp16 mode, kv_cache int8 quantization can be enabled, and a single card can serve more users.
First execute the quantization script, and the quantization parameters are stored in the weight directory transformed by `deploy.py`.

```
python3 -m lmdeploy.lite.apis.kv_qparams \
  --model $HF_MODEL \
  --output_dir $DEPLOY_WEIGHT_DIR \
  --symmetry True \   # Whether to use symmetric or asymmetric quantization.
  --offload  False \  # Whether to offload some modules to CPU to save GPU memory.
  --num_tp 1 \   # The number of GPUs used for tensor parallelism
```

Then adjust `config.ini`

- `use_context_fmha` changed to 0, means off
- `quant_policy` is set to 4. This parameter defaults to 0, which means it is not enabled

Here is [quantization test results](./docs/zh_cn/quantization.md).

## Contributing

We appreciate all contributions to LMDeploy. Please refer to [CONTRIBUTING.md](.github/CONTRIBUTING.md) for the contributing guideline.

## Acknowledgement

- [FasterTransformer](https://github.com/NVIDIA/FasterTransformer)

## License

This project is released under the [Apache 2.0 license](LICENSE).
