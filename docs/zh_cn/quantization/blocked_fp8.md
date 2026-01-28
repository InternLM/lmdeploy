# Blocked FP8 模型量化

LMDeploy 支持一种仅权重的 (weight-only) Blocked FP8 量化方法。该方法将模型的权重以分块（blocked）的形式量化为 8-bit 浮点数，可以在支持的硬件上保持良好性能的同时，有效降低模型的显存占用。

在进行量化和推理之前，请确保按照[安装指南](../get_started/installation.md)安装了 lmdeploy。

```shell
pip install lmdeploy[all]
```

## 模型量化

仅需执行一条命令，就可以完成模型量化工作。该脚本会加载模型，将线性层量化为 Blocked FP8 格式，并将最终的模型和配置文件保存在指定的工作目录中。

使用的命令是 `lmdeploy lite blocked_fp8`。

以下是如何量化 `OpenGVLab/InternVL3_5-8B` 的示例：

```shell
export HF_MODEL=OpenGVLab/InternVL3_5-8B
export WORK_DIR=OpenGVLab/InternVL3_5-8B-FP8

lmdeploy lite blocked_fp8 $HF_MODEL \
  --work-dir $WORK_DIR \
  --quant-dtype fp8 \
  --block-size 128
```

命令行的主要参数说明：

- `--work-dir`: 用于保存量化后的模型权重和配置的工作目录。
- `--quant-dtype`: 目标 FP8 格式。可以是 `float8_e4m3fn` (与传入 'fp8' 效果相同，推荐) 或 `float8_e5m2`。
- `--block-size`: 量化的块大小。默认值 `128` 通常是一个不错的选择。

## 模型推理

您可以使用 `turbomind` 和 `pytorch` 后端对量化后的模型进行批量离线推理。

这是一个简单的代码示例：

```python
from lmdeploy import pipeline

pipe = pipeline("OpenGVLab/InternVL3_5-8B-FP8")
response = pipe(["Hi, pls intro yourself", "Shanghai is"])
print(response)
```

## 推理服务

LMDeploy 的 `api_server` 可用于服务化部署 Blocked FP8 模型。

```shell
lmdeploy serve api_server OpenGVLab/InternVL3_5-8B-FP8
```

服务的默认端口是 `23333`。

您可以通过 Swagger UI `http://0.0.0.0:23333` 在线阅读和试用 `api_server` 的各接口，也可直接查阅[文档](../llm/api_server.md)，了解各接口的定义和使用方法。
