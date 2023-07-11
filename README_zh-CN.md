<div align="center">
  <img src="resources/lmdeploy-logo.png" width="450"/>

[English](README.md) | 简体中文

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

## 简介

LMDeploy 由 [MMDeploy](https://github.com/open-mmlab/mmdeploy) 和 [MMRazor](https://github.com/open-mmlab/mmrazor) 团队联合开发，是涵盖了 LLM 任务的全套轻量化、部署和服务解决方案。
这个强大的工具箱提供以下核心功能：

- **高效推理引擎 TurboMind**：基于 [FasterTransformer](https://github.com/NVIDIA/FasterTransformer)，我们实现了高效推理引擎 TurboMind，支持 InternLM、LLaMA、vicuna等模型在 NVIDIA GPU 上的推理。

- **交互推理方式**：通过缓存多轮对话过程中 attention 的 k/v，记住对话历史，从而避免重复处理历史会话。

- **多 GPU 部署和量化**：我们提供了全面的模型部署和量化支持，已在不同规模上完成验证。

- **persistent batch 推理**：进一步优化模型执行效率。

  ![PersistentBatchInference](https://github.com/InternLM/lmdeploy/assets/67539920/e3876167-0671-44fc-ac52-5a0f9382493e)

## 性能

如下图所示，我们对比了 facebookresearch/llama、HuggingFace Transformers、DeepSpeed 在 7B 模型上的token生成的速度。

测试设备：NVIDIA A100(80G)

测试指标：吞吐量（token/s)

测试数据：输入token数为1，生成token数为2048

TurboMind 的吞吐量超过 2000 token/s, 整体比 DeepSpeed 提升约 5% - 15%，比 huggingface transformers 提升 2.3 倍

![benchmark](https://user-images.githubusercontent.com/12756472/251422522-e94a3db9-eb16-432a-8d8c-078945e7b99a.png)

## 快速上手

### 安装

```shell
conda create -n lmdeploy python=3.10
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
git clone https://huggingface.co/internlm/internlm-7b /path/to/internlm-7b

# if you want to clone without large files – just their pointers
# prepend your git clone with the following env var:
GIT_LFS_SKIP_SMUDGE=1

# 2. 转换为 trubomind 要求的格式。默认存放路径为 ./workspace
python3 -m lmdeploy.serve.turbomind.deploy internlm-7b /path/to/internlm-7b hf

```

#### 使用 turbomind 推理

```shell
docker run --gpus all --rm -v $(pwd)/workspace:/workspace -it openmmlab/lmdeploy:latest \
    python3 -m lmdeploy.turbomind.chat internlm /workspace
```

```{note}
turbomind 在使用 FP16 精度推理 InternLM-7B 模型时，显存开销至少需要 22.7G。建议使用 3090, V100，A100等型号的显卡
```

#### 部署推理服务

使用下面的命令启动推理服务：

```shell
bash workspace/service_docker_up.sh
```

你可以通过命令行方式与推理服务进行对话：

```shell
python3 lmdeploy.serve.client {server_ip_addresss}:33337 internlm
```

也可以通过 WebUI 方式来对话：

```
python3 lmdeploy.app {server_ip_addresss}:33337 internlm
```

![](https://github.com/InternLM/lmdeploy/assets/67539920/08d1e6f2-3767-44d5-8654-c85767cec2ab)

其他模型的部署方式，比如 LLaMA，vicuna，请参考[这里](docs/zh_cn/serving.md)

## 量化部署

在 fp16 模式下，可以开启 kv_cache int8 量化，单卡可服务更多用户。
首先执行量化脚本，量化参数存放到 `deploy.py` 转换的 weight 目录下。

```
python3 -m lmdeploy.lite.apis.kv_qparams \
  --model $HF_MODEL \
  --output_dir $DEPLOY_WEIGHT_DIR \
  --symmetry True \ # 对称量化或非对称量化，默认为 True
  --offload  False \ # 将模型放在 CPU，只在推理时加载部分模块到 GPU，默认为 False
  --num_tp 1  \  # Tensor 并行使用的 GPU 数，和 deploy.py 保持一致
```

然后调整 `config.ini`

- `use_context_fmha` 改为 0，表示关闭
- `quant_policy` 设置为 4。此参数默认为 0，表示不开启

这里是[量化测试结果](./docs/zh_cn/quantization.md)。

## 贡献指南

我们感谢所有的贡献者为改进和提升 LMDeploy 所作出的努力。请参考[贡献指南](.github/CONTRIBUTING.md)来了解参与项目贡献的相关指引。

## 致谢

- [FasterTransformer](https://github.com/NVIDIA/FasterTransformer)

## License

该项目采用 [Apache 2.0 开源许可证](LICENSE)。
