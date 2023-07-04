<div align="center">
  <img src="resources/lmdeploy-logo.png" width="450"/>
  <div>&nbsp;</div>
  <div align="center">
    <b><font size="5">OpenMMLab website</font></b>
    <sup>
        <a href="https://openmmlab.com">
        <i><font size="4">HOT</font></i>
      </a>
    </sup>
    &nbsp;&nbsp;&nbsp;&nbsp;
    <b><font size="5">OpenMMLab platform</font></b>
    <sup>
      <a href="https://platform.openmmlab.com">
        <i><font size="4">TRY IT OUT</font></i>
      </a>
    </sup>
  </div>
  <div>&nbsp;</div>

[![docs](https://img.shields.io/badge/docs-latest-blue)](https://lmdeploy.readthedocs.io/en/latest/)
[![codecov](https://codecov.io/gh/open-mmlab/lmdeploy/branch/main/graph/badge.svg)](https://codecov.io/gh/open-mmlab/lmdeploy)
[![license](https://img.shields.io/github/license/open-mmlab/lmdeploy.svg)](https://github.com/open-mmlab/mmdeploy/tree/main/LICENSE)
[![issue resolution](https://img.shields.io/github/issues-closed-raw/open-mmlab/lmdeploy)](https://github.com/open-mmlab/lmdeploy/issues)
[![open issues](https://img.shields.io/github/issues-raw/open-mmlab/lmdeploy)](https://github.com/open-mmlab/lmdeploy/issues)

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

- **高效推理引擎 TurboMind**：基于 [FasterTransformer](https://github.com/NVIDIA/FasterTransformer)，我们实现了高效推理引擎 TurboMind，它支持 LLaMA 及其变体模型在 NVIDIA GPU 上的推理。

- **交互推理方式**：通过缓存多轮对话过程中 attention 的 k/v，记住对话历史，从而避免重复处理历史会话。

  <div align="center">
    <img src="https://github.com/NVIDIA/FasterTransformer/blob/main/docs/images/gpt/gpt_interactive_generation.2.png?raw=true" width="600"/>
  </div>

- **persistent batch 推理**：进一步优化模型执行效率。

  ![PersistentBatchInference](https://github.com/open-mmlab/lmdeploy/assets/25839884/8f8b57b8-42af-4b71-ad74-e75f39b10694)

- **多 GPU 部署和量化**：我们提供了全面的模型部署和量化支持，已经在 7～100B 模型上完成验证。


## 快速上手

### 安装

```shell
conda create -n open-mmlab python=3.8
conda activate open-mmlab
git clone https://github.com/open-mmlab/lmdeploy.git
cd lmdeploy
pip install -e .
```

### 编译

下载 docker image `openmmlab/lmdeploy:latest`，挂载 lmdeploy 的数据卷，启动 container，在 container 内执行以下命令：

```shell
mkdir build && cd build
../generate.sh
make -j$(nproc) && make install
```

### 部署 [LLaMA](https://github.com/facebookresearch/llama) 服务

请填写[这张表](https://docs.google.com/forms/d/e/1FAIpQLSfqNECQnMkycAp2jP4Z9TFX0cGR4uf7b_fBxjY_OjhJILlKGA/viewform)，获取 LLaMA 模型权重。

执行如下命令，可以把 LLaMA 模型部署到 NVIDIA GPU Server：

<details open>
<summary><b>7B</b></summary>

```shell
python3 lmdeploy/serve/turbomind/deploy.py llama-7B /path/to/llama-7b llama \
    --tokenizer_path /path/to/tokenizer/model
bash workspace/service_docker_up.sh --lib-dir $(pwd)/build/install/backends/turbomind
```

</details>

<details open>
<summary><b>13B</b></summary>

```shell
python3 lmdeploy/serve/turbomind/deploy.py llama-13B /path/to/llama-13b llama \
    --tokenizer_path /path/to/tokenizer/model --tp 2
bash workspace/service_docker_up.sh --lib-dir $(pwd)/build/install/backends/turbomind
```

</details>

### 部署 [Vicuna](https://lmsys.org/blog/2023-03-30-vicuna/) 服务

<details open>
<summary><b>7B</b></summary>

```shell
python3 -m pip install fschat
python3 -m fastchat.model.apply_delta \
  --base-model-path /path/to/llama-7b \
  --target-model-path /path/to/vicuna-7b \
  --delta-path lmsys/vicuna-7b-delta-v1.1

python3 lmdeploy/serve/turbomind/deploy.py vicuna-7B /path/to/vicuna-7b hf
bash workspace/service_docker_up.sh --lib-dir $(pwd)/build/install/backends/turbomind
```

</details>

<details>
<summary><b>13B</b></summary>

```shell
python3 -m pip install fschat
python3 -m fastchat.model.apply_delta \
  --base-model-path /path/to/llama-13b \
  --target-model-path /path/to/vicuna-13b \
  --delta-path lmsys/vicuna-13b-delta-v1.1

python3 lmdeploy/serve/turbomind/deploy.py vicuna-13B /path/to/vicuna-13b hf
bash workspace/service_docker_up.sh --lib-dir $(pwd)/build/install/backends/turbomind
```

</details>

## 通过命令行推理

```shell
python3 lmdeploy/serve/client.py {server_ip_addresss}:33337
```

## 使用浏览器推理

```shell
python3 lmdeploy/app.py {server_ip_addresss}:33337 {model_name}
```

## 量化部署

在 fp16 模式下，可以开启 kv_cache int8 量化，单卡可服务更多用户。
首先执行量化脚本，量化参数存放到 `deploy.py` 转换的 weight 目录下。
然后调整 `config.ini`

- `use_context_fmha` 改为 0，表示关闭
- `quant_policy` 设置为 4。此参数默认为 0，表示不开启

## 贡献指南

我们感谢所有的贡献者为改进和提升 LMDeploy 所作出的努力。请参考[贡献指南](.github/CONTRIBUTING.md)来了解参与项目贡献的相关指引。

## 致谢

- [FasterTransformer](https://github.com/NVIDIA/FasterTransformer)

## License

该项目采用 [Apache 2.0 开源许可证](LICENSE)。
