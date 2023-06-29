<div align="center">
  <img src="resources/llmdeploy-logo.png" width="450"/>
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

[![docs](https://img.shields.io/badge/docs-latest-blue)](https://llmdeploy.readthedocs.io/en/latest/)
[![codecov](https://codecov.io/gh/open-mmlab/llmdeploy/branch/main/graph/badge.svg)](https://codecov.io/gh/open-mmlab/llmdeploy)
[![license](https://img.shields.io/github/license/open-mmlab/llmdeploy.svg)](https://github.com/open-mmlab/mmdeploy/tree/main/LICENSE)
[![issue resolution](https://img.shields.io/github/issues-closed-raw/open-mmlab/llmdeploy)](https://github.com/open-mmlab/llmdeploy/issues)
[![open issues](https://img.shields.io/github/issues-raw/open-mmlab/llmdeploy)](https://github.com/open-mmlab/llmdeploy/issues)

[English](README.md) | 简体中文

</div>

<div align="center">
  <a href="https://openmmlab.medium.com/" style="text-decoration:none;">
    <img src="https://user-images.githubusercontent.com/25839884/218352562-cdded397-b0f3-4ca1-b8dd-a60df8dca75b.png" width="3%" alt="" /></a>
  <img src="https://user-images.githubusercontent.com/25839884/218346358-56cc8e2f-a2b8-487f-9088-32480cceabcf.png" width="3%" alt="" />
  <a href="https://discord.gg/raweFPmdzG" style="text-decoration:none;">
    <img src="https://user-images.githubusercontent.com/25839884/218347213-c080267f-cbb6-443e-8532-8e1ed9a58ea9.png" width="3%" alt="" /></a>
  <img src="https://user-images.githubusercontent.com/25839884/218346358-56cc8e2f-a2b8-487f-9088-32480cceabcf.png" width="3%" alt="" />
  <a href="https://twitter.com/OpenMMLab" style="text-decoration:none;">
    <img src="https://user-images.githubusercontent.com/25839884/218346637-d30c8a0f-3eba-4699-8131-512fb06d46db.png" width="3%" alt="" /></a>
  <img src="https://user-images.githubusercontent.com/25839884/218346358-56cc8e2f-a2b8-487f-9088-32480cceabcf.png" width="3%" alt="" />
  <a href="https://www.youtube.com/openmmlab" style="text-decoration:none;">
    <img src="https://user-images.githubusercontent.com/25839884/218346691-ceb2116a-465a-40af-8424-9f30d2348ca9.png" width="3%" alt="" /></a>
</div>

## 简介

## 安装

```shell
conda create -n open-mmlab python=3.8
conda activate open-mmlab
git clone https://github.com/open-mmlab/llmdeploy.git
cd llmdeploy
pip install -e .
```

## 快速上手

### 部署 [LLaMA](https://github.com/facebookresearch/llama) 服务

请填写[这张表](https://docs.google.com/forms/d/e/1FAIpQLSfqNECQnMkycAp2jP4Z9TFX0cGR4uf7b_fBxjY_OjhJILlKGA/viewform)，获取 LLaMA 模型权重。

执行下面任一命令，可以把 LLaMA 模型部署到 NVIDIA GPU Server：

<details open>
<summary><b>7B</b></summary>

```shell
python3 llmdeploy/serve/fastertransformer/deploy.py llama-7B /path/to/llama-7b llama \
    --tokenizer_path /path/to/tokenizer/model
bash workspace/service_docker_up.sh
```

</details>

<details open>
<summary><b>13B</b></summary>

```shell
python3 llmdeploy/serve/fastertransformer/deploy.py llama-13B /path/to/llama-13b llama \
    --tokenizer_path /path/to/tokenizer/model --tp 2
bash workspace/service_docker_up.sh
```

</details>

<details open>
<summary><b>33B</b></summary>

```shell
python3 llmdeploy/serve/fastertransformer/deploy.py llama-33B /path/to/llama-33b llama \
    --tokenizer_path /path/to/tokenizer/model --tp 4
bash workspace/service_docker_up.sh
```

</details>

<details open>
<summary><b>65B</b></summary>

```shell
python3 llmdeploy/serve/fastertransformer/deploy.py llama-65B /path/to/llama-65b llama \
    --tokenizer_path /path/to/tokenizer/model --tp 8
bash workspace/service_docker_up.sh
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

python3 llmdeploy/serve/fastertransformer/deploy.py vicuna-7B /path/to/vicuna-7b hf
bash workspace/service_docker_up.sh
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

python3 llmdeploy/serve/fastertransformer/deploy.py vicuna-13B /path/to/vicuna-13b hf
bash workspace/service_docker_up.sh
```

</details>

## 通过命令行推理

```shell
python3 llmdeploy/serve/client.py {server_ip_addresss}:33337 1
```

## 使用浏览器推理

```shell
python3 llmdeploy/webui/app.py {server_ip_addresss}:33337 model_name
```

## 贡献指南

我们感谢所有的贡献者为改进和提升 LLMDeploy 所作出的努力。请参考[贡献指南](.github/CONTRIBUTING.md)来了解参与项目贡献的相关指引。

## 致谢

- [FasterTransformer](https://github.com/NVIDIA/FasterTransformer)

## License

该项目采用 [Apache 2.0 开源许可证](LICENSE)。
