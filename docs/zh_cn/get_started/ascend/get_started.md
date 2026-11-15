# 华为昇腾

我们基于 LMDeploy 的 PytorchEngine，增加了华为昇腾设备的支持，目前支持的型号是**Atlas 800T A3，Atlas 800T A2和Atlas 300I Duo**。在华为昇腾上使用 LMDeploy 的方法与在英伟达 GPU 上使用 PytorchEngine 后端的方法几乎相同。在阅读本教程之前，请先阅读原版的[快速开始](../get_started.md)。

支持的模型列表在[这里](../../supported_models/supported_models.md#pytorchengine-其他平台).

## A3 原生后端安装

当前 A3 PyTorch 后端由 LMDeploy 内部实现，无需 dlinfer。请使用未安装 dlinfer 的独立
Python 环境，并先加载当前安装的 CANN 和 NNAL/ATB 环境脚本。以下源码安装示例沿用
[`Dockerfile_ascend_a3`](../../../../docker/Dockerfile_ascend_a3) 的 PyTorch 版本：

```bash
python3.11 -m venv .venv-ascend
source .venv-ascend/bin/activate
python -m pip install torch==2.9.0 torch-npu==2.9.0 torchvision==0.24.0
python -m pip install -r requirements/ascend_kernels.txt
LMDEPLOY_TARGET_DEVICE=ascend python -m pip install .
python tools/ascend/check_runtime.py --require-kernels
```

在仓库根目录执行这些命令。CANN、torch_npu、Triton Ascend 和外部
`triton-ascend-kernels` 仍是运行依赖；请将后者固定在与环境兼容的 revision。
推理入口继续使用 `device_type="ascend"` 或 `--device ascend`。图模式通过
`torch.npu.NPUGraph` 执行，可使用 `eager_mode=True` 或 `--eager-mode` 关闭。

下方预构建镜像说明对应依赖 dlinfer 的旧版本。本次原生迁移面向 A3，A2 和 Atlas 300I Duo
尚未在此后端上验证。使用容器时，可通过 `docker/Dockerfile_ascend_a3` 构建当前源码，
其 CANN/PyTorch 组合需要与宿主驱动及运行时匹配。

## 旧版预构建镜像

> \[!IMPORTANT\]
> 我们已经在阿里云上提供了构建完成的鲲鹏CPU版本的镜像。
> 请使用下面的命令来拉取镜像:
>
> Atlas 800T A3:
> `docker pull crpi-4crprmm5baj1v8iv.cn-hangzhou.personal.cr.aliyuncs.com/lmdeploy_dlinfer/ascend:a3-latest`
> （Atlas 800T A3目前只支持Qwen系列的算子模式下运行）
>
> Atlas 800T A2:
> `docker pull crpi-4crprmm5baj1v8iv.cn-hangzhou.personal.cr.aliyuncs.com/lmdeploy_dlinfer/ascend:a2-latest`
>
> Atlas 300I Duo:
> `docker pull crpi-4crprmm5baj1v8iv.cn-hangzhou.personal.cr.aliyuncs.com/lmdeploy_dlinfer/ascend:300i-duo-latest`
> （Atlas 300I Duo目前只支持非eager模式）
>
> 如果您希望自己构建环境，请参考[这里](../../../../docker)的dockerfile来自己构建。

## 离线批处理

### LLM 推理

将`device_type="ascend"`加入`PytorchEngineConfig`的参数中。

```python
from lmdeploy import pipeline
from lmdeploy import PytorchEngineConfig
pipe = pipeline("internlm/internlm2_5-7b-chat",
             backend_config=PytorchEngineConfig(tp=1, device_type="ascend"))
question = ["Shanghai is", "Please introduce China", "How are you?"]
response = pipe(question)
print(response)
```

### VLM 推理

将`device_type="ascend"`加入`PytorchEngineConfig`的参数中。

```python
from lmdeploy import pipeline, PytorchEngineConfig
from lmdeploy.vl import load_image
pipe = pipeline('OpenGVLab/InternVL2-2B',
     backend_config=PytorchEngineConfig(tp=1, device_type='ascend'))
image = load_image('https://raw.githubusercontent.com/open-mmlab/mmdeploy/main/tests/data/tiger.jpeg')
response = pipe(('describe this image', image))
print(response)
```

## 在线服务

### LLM 模型服务

将`--device ascend`加入到服务启动命令中。

```bash
lmdeploy serve api_server --backend pytorch --device ascend internlm/internlm2_5-7b-chat
```

也可以运行以下命令启动容器运行LLM模型服务。

```bash
docker run -it --net=host crpi-4crprmm5baj1v8iv.cn-hangzhou.personal.cr.aliyuncs.com/lmdeploy_dlinfer/ascend:a2-latest \
    bash -i -c "lmdeploy serve api_server --backend pytorch --device ascend internlm/internlm2_5-7b-chat"
```

### VLM 模型服务

将`--device ascend`加入到服务启动命令中。

```bash
lmdeploy serve api_server --backend pytorch --device ascend OpenGVLab/InternVL2-2B
```

也可以运行以下命令启动容器运行VLM模型服务。

```bash
docker run -it --net=host crpi-4crprmm5baj1v8iv.cn-hangzhou.personal.cr.aliyuncs.com/lmdeploy_dlinfer/ascend:a2-latest \
    bash -i -c "lmdeploy serve api_server --backend pytorch --device ascend OpenGVLab/InternVL2-2B"
```

## 使用命令行与LLM模型对话

将`--device ascend`加入到服务启动命令中。

```bash
lmdeploy chat internlm/internlm2_5-7b-chat --backend pytorch --device ascend
```

也可以运行以下命令使启动容器后开启lmdeploy聊天

```bash
docker run -it crpi-4crprmm5baj1v8iv.cn-hangzhou.personal.cr.aliyuncs.com/lmdeploy_dlinfer/ascend:a2-latest \
    bash -i -c "lmdeploy chat --backend pytorch --device ascend internlm/internlm2_5-7b-chat"
```

## 量化

### w4a16 AWQ

运行下面的代码可以在Atlas 800T A2上对权重进行W4A16量化。

```bash
lmdeploy lite auto_awq $HF_MODEL --work-dir $WORK_DIR --device npu
```

支持的模型列表请参考[支持的模型](../../supported_models/supported_models.md)。

### w8a8 SMOOTH_QUANT

运行下面的代码可以在Atlas 800T A2上对权重进行W8A8量化。

```bash
lmdeploy lite smooth_quant $HF_MODEL --work-dir $WORK_DIR --device npu
```

支持的模型列表请参考[支持的模型](../../supported_models/supported_models.md)。

### int8 KV-cache 量化

昇腾后端现在支持了在eager模式下的离线int8 KV-cache量化。

详细使用方式请请参考这篇[文章](https://github.com/DeepLink-org/dlinfer/blob/main/docs/quant/ascend_kv_quant.md)。

## Atlas 300I Duo上的限制

1. 只支持dtype=float16。
2. 只支持图模式，请不要加上--eager-mode。
