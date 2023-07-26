<div align="center">
  <img src="resources/lmdeploy-logo.png" width="450"/>

[English](README.md) | 简体中文

</div>

<p align="center">
    👋 join us on <a href="https://discord.gg/xa29JuW87d" target="_blank">Discord</a> and <a href="https://r.vansin.top/?r=internwx" target="_blank">WeChat</a>
</p>

______________________________________________________________________

## 更新 🎉

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

## 性能

**场景一**: 固定的输入、输出token数（1,2048），测试 output token throughput

**场景二**: 使用真实数据，测试 request throughput

测试配置：LLaMA-7B, NVIDIA A100(80G)

TurboMind 的 output token throughput 超过 2000 token/s, 整体比 DeepSpeed 提升约 5% - 15%，比 huggingface transformers 提升 2.3 倍
在 request throughput 指标上，TurboMind 的效率比 vLLM 高 30%

![benchmark](https://github.com/InternLM/lmdeploy/assets/4560679/7775c518-608e-4e5b-be73-7645a444e774)

## 快速上手

### 安装

```shell
conda create -n lmdeploy python=3.10 -y
conda activate lmdeploy
git clone https://github.com/InternLM/lmdeploy.git
cd lmdeploy
pip install -e .
```

### 部署 InternLM

#### 获取 InternLM 模型

```shell
# 1. 下载 InternLM 模型

# Make sure you have git-lfs installed (https://git-lfs.com)
git lfs install
git clone https://huggingface.co/internlm/internlm-chat-7b /path/to/internlm-chat-7b

# if you want to clone without large files – just their pointers
# prepend your git clone with the following env var:
GIT_LFS_SKIP_SMUDGE=1

# 2. 转换为 trubomind 要求的格式。默认存放路径为 ./workspace
python3 -m lmdeploy.serve.turbomind.deploy internlm-chat-7b /path/to/internlm-chat-7b

```

#### 使用 turbomind 推理

```shell
docker run --gpus all --rm -v $(pwd)/workspace:/workspace -it openmmlab/lmdeploy:latest \
    python3 -m lmdeploy.turbomind.chat /workspace
```

> **Note**
> turbomind 在使用 FP16 精度推理 InternLM-7B 模型时，显存开销至少需要 15.7G。建议使用 3090, V100，A100等型号的显卡

> **Note**
> 使用 Tensor 并发可以利用多张 GPU 进行推理。在 `chat` 时添加参数 `--tp=<num_gpu>` 可以启动运行时 TP。

#### 部署推理服务

使用下面的命令启动推理服务：

```shell
bash workspace/service_docker_up.sh
```

你可以通过命令行方式与推理服务进行对话：

```shell
python3 -m lmdeploy.serve.client {server_ip_addresss}:33337
```

也可以通过 WebUI 方式来对话：

```shell
python3 -m lmdeploy.app {server_ip_addresss}:33337
```

![](https://github.com/InternLM/lmdeploy/assets/67539920/08d1e6f2-3767-44d5-8654-c85767cec2ab)

其他模型的部署方式，比如 LLaMA，LLaMA-2，vicuna等等，请参考[这里](docs/zh_cn/serving.md)

### 基于 PyTorch 的推理

你必须确保环境中有安装 deepspeed：

```
pip install deepspeed
```

#### 单个 GPU

```shell
python3 -m lmdeploy.pytorch.chat $NAME_OR_PATH_TO_HF_MODEL\
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

在 fp16 模式下，可以开启 kv_cache int8 量化，单卡可服务更多用户。
首先执行量化脚本，量化参数存放到 `deploy.py` 转换的 `workspace/triton_models/weights` 目录下。

```
python3 -m lmdeploy.lite.apis.kv_qparams \
  --model $HF_MODEL \
  --output_dir $DEPLOY_WEIGHT_DIR \
  --symmetry True \ # 对称量化或非对称量化，默认为 True
  --offload  False \ # 将模型放在 CPU，只在推理时加载部分模块到 GPU，默认为 False
  --num_tp 1  \  # Tensor 并行使用的 GPU 数，和 deploy.py 保持一致
```

然后调整 `workspace/triton_models/weights/config.ini`

- `use_context_fmha` 改为 0，表示关闭
- `quant_policy` 设置为 4。此参数默认为 0，表示不开启

这里是[量化测试结果](./docs/zh_cn/quantization.md)。

> **Warning**
> 量化部署不支持运行时 Tensor 并发。如果希望使用 Tensor 并发，需要在 deploy 时配置 tp 参数。

## 贡献指南

我们感谢所有的贡献者为改进和提升 LMDeploy 所作出的努力。请参考[贡献指南](.github/CONTRIBUTING.md)来了解参与项目贡献的相关指引。

## 致谢

- [FasterTransformer](https://github.com/NVIDIA/FasterTransformer)

## License

该项目采用 [Apache 2.0 开源许可证](LICENSE)。
