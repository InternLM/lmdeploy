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

### GDN/MTP 状态正确性与编译器升级

circular GDN 状态路径默认使用 `num_stages=1`，其他 recurrent 路径保留
`num_stages=3`。AscendNPU-IR 1.2.0 的 `042c923d` 构建在自动多缓冲时可能写错
FP32 中间状态，而当前输出仍然正确。MTP 后续读取这些状态后，贪婪采样的 target
logits 和生成 token 会在重复运行间出现差异。单缓冲保护受影响编译器上的这条路径，
新编译器也沿用这一默认值；正确的多缓冲并非在所有已测形状下都更快。

上游修复 [`05f2adcf`](https://github.com/Ascend/AscendNPU-IR/commit/05f2adcf3db366377f7c5e6f5bb9bd3c61c36b3e)
已进入 AscendNPU-IR 2.0.0（`efb71415`），由官方 Triton-Ascend 3.2.2.post2 wheel
捆绑交付。已有兼容 A3 原生运行环境可以选择以下升级配置：

```bash
python -m pip install --no-deps -r requirements/ascend_triton_npuir2.txt \
    --extra-index-url https://mirrors.huaweicloud.com/ascend/repos/pypi
```

请在拥有独立包文件的隔离环境执行。此配置只更新 Triton 及其编译器组件，保留已有
运行依赖；不要与 `requirements/ascend_kernels.txt` 中的旧 Triton 版本约束在同一次
安装中混用。实测升级环境为 Python 3.12.13、Torch 2.10.0+cpu、torch_npu
2.10.0.post4、CANN 9.1.0。这是定向编译器更新的验证，不等于全新依赖解析安装，
也不等于现有 CANN 8.5 Docker 配方与新编译器的组合已经通过。
[官方推荐组合](https://ascendnpu-ir.gitcode.com/en/introduction/quick_start/version_compatibility.html)
为 NPUIR 2.0.0 与 CANN 9.2.0；该完整运行时/驱动升级尚未实测。

Triton 优先选择自己捆绑的编译器。升级后应检查实际执行路径和构建号，仅查看
`triton.__version__` 不足以确认：

```python
import subprocess
from triton.backends.ascend.utils import _get_npucompiler_path

compiler, compiler_env = _get_npucompiler_path()
print(compiler)
subprocess.run([compiler, '--version'], env=compiler_env, check=True)
```

在 A3 上执行状态回归：

```bash
python -m pytest -q --run-ascend-hardware \
    tests/pytorch/ascend/hardware/test_gdn_state_writeback.py
```

测试覆盖 Q=1/2/4 的 eager/graph 模式，每次重放恢复初始状态，并将所有保存的状态与
CPU 递推参考比较。[完整调查报告](https://github.com/silencelamb/lmdeploy/blob/docs/ascend-a3-validation-20261114/docs/zh_cn/advance/ascend_a3_validation/mtp_ray_investigation_20261116.md)
记录了编译器 IR、固定版本、MTP 对照、随形状变化的性能和未覆盖范围。算子及 target
主干计时不能替代端到端服务吞吐验收。

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
