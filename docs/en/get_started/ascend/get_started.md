# Get Started with Huawei Ascend

We currently support running lmdeploy on **Atlas 800T A3, Atlas 800T A2 and Atlas 300I Duo**.
The usage of lmdeploy on a Huawei Ascend device is almost the same as its usage on CUDA with PytorchEngine in lmdeploy.
Please read the original [Get Started](../get_started.md) guide before reading this tutorial.

Here is the [supported model list](../../supported_models/supported_models.md#pytorchengine-on-other-platforms).

## Native A3 installation

The current A3 PyTorch backend is implemented inside LMDeploy and does not
require dlinfer. Use a separate Python environment without dlinfer installed.
Load the CANN and NNAL/ATB environment scripts for your installation first.
The following source-install recipe uses the PyTorch versions in
[`Dockerfile_ascend_a3`](../../../../docker/Dockerfile_ascend_a3):

```bash
python3.11 -m venv .venv-ascend
source .venv-ascend/bin/activate
python -m pip install torch==2.9.0 torch-npu==2.9.0 torchvision==0.24.0
python -m pip install -r requirements/ascend_kernels.txt
LMDEPLOY_TARGET_DEVICE=ascend python -m pip install .
python tools/ascend/check_runtime.py --require-kernels
```

Run these commands from the repository root. CANN, torch_npu, Triton Ascend,
and the external `triton-ascend-kernels` library remain required; pin the latter
to a revision compatible with your environment. Keep `device_type="ascend"`
or `--device ascend` in inference commands. Graph mode uses `torch.npu.NPUGraph`;
use `eager_mode=True` or `--eager-mode` to disable it.

The prebuilt images below describe older dlinfer-based releases. The native
migration targets A3; A2 and Atlas 300I Duo have not been validated with this
backend. Build the current A3 checkout with `docker/Dockerfile_ascend_a3` when
using containers. Its CANN/PyTorch recipe must match the host driver and runtime.

## Legacy prebuilt images

> \[!IMPORTANT\]
> We have uploaded a docker image with KUNPENG CPU to aliyun.
> Please try to pull the image by following command:
>
> Atlas 800T A3:
>
> `docker pull crpi-4crprmm5baj1v8iv.cn-hangzhou.personal.cr.aliyuncs.com/lmdeploy_dlinfer/ascend:a3-latest`
>
> (Atlas 800T A3 currently supports only the Qwen-series with eager mode.)
>
> Atlas 800T A2:
>
> `docker pull crpi-4crprmm5baj1v8iv.cn-hangzhou.personal.cr.aliyuncs.com/lmdeploy_dlinfer/ascend:a2-latest`
>
> 300I Duo:
>
> `docker pull crpi-4crprmm5baj1v8iv.cn-hangzhou.personal.cr.aliyuncs.com/lmdeploy_dlinfer/ascend:300i-duo-latest`
>
> (Atlas 300I Duo currently works only with graph mode.)
>
> To build the environment yourself, refer to the Dockerfiles [here](../../../../docker).

## Offline batch inference

### LLM inference

Set `device_type="ascend"` in the `PytorchEngineConfig`:

```python
from lmdeploy import pipeline
from lmdeploy import PytorchEngineConfig
pipe = pipeline("internlm/internlm2_5-7b-chat",
        backend_config=PytorchEngineConfig(tp=1, device_type="ascend"))
question = ["Shanghai is", "Please introduce China", "How are you?"]
response = pipe(question)
print(response)
```

### VLM inference

Set `device_type="ascend"` in the `PytorchEngineConfig`:

```python
from lmdeploy import pipeline, PytorchEngineConfig
from lmdeploy.vl import load_image
pipe = pipeline('OpenGVLab/InternVL2-2B',
        backend_config=PytorchEngineConfig(tp=1, device_type='ascend'))
image = load_image('https://raw.githubusercontent.com/open-mmlab/mmdeploy/main/tests/data/tiger.jpeg')
response = pipe(('describe this image', image))
print(response)
```

## Online serving

### Serve a LLM model

Add `--device ascend` in the serve command.

```bash
lmdeploy serve api_server --backend pytorch --device ascend internlm/internlm2_5-7b-chat
```

Run the following commands to launch docker container for lmdeploy LLM serving:

```bash
docker run -it --net=host crpi-4crprmm5baj1v8iv.cn-hangzhou.personal.cr.aliyuncs.com/lmdeploy_dlinfer/ascend:a2-latest \
    bash -i -c "lmdeploy serve api_server --backend pytorch --device ascend internlm/internlm2_5-7b-chat"
```

### Serve a VLM model

Add `--device ascend` in the serve command

```bash
lmdeploy serve api_server --backend pytorch --device ascend OpenGVLab/InternVL2-2B
```

Run the following commands to launch docker container for lmdeploy VLM serving:

```bash
docker run -it --net=host crpi-4crprmm5baj1v8iv.cn-hangzhou.personal.cr.aliyuncs.com/lmdeploy_dlinfer/ascend:a2-latest \
    bash -i -c "lmdeploy serve api_server --backend pytorch --device ascend OpenGVLab/InternVL2-2B"
```

## Inference with Command line Interface

Add `--device ascend` in the serve command.

```bash
lmdeploy chat internlm/internlm2_5-7b-chat --backend pytorch --device ascend
```

Run the following commands to launch lmdeploy chatting after starting container:

```bash
docker run -it crpi-4crprmm5baj1v8iv.cn-hangzhou.personal.cr.aliyuncs.com/lmdeploy_dlinfer/ascend:a2-latest \
    bash -i -c "lmdeploy chat --backend pytorch --device ascend internlm/internlm2_5-7b-chat"
```

## Quantization

### w4a16 AWQ

Run the following commands to quantize weights on Atlas 800T A2.

```bash
lmdeploy lite auto_awq $HF_MODEL --work-dir $WORK_DIR --device npu
```

Please check [supported_models](../../supported_models/supported_models.md) before use this feature.

### w8a8 SMOOTH_QUANT

Run the following commands to quantize weights on Atlas 800T A2.

```bash
lmdeploy lite smooth_quant $HF_MODEL --work-dir $WORK_DIR --device npu
```

Please check [supported_models](../../supported_models/supported_models.md) before use this feature.

### int8 KV-cache Quantization

Ascend backend has supported offline int8 KV-cache Quantization on eager mode.

Please refer this [doc](https://github.com/DeepLink-org/dlinfer/blob/main/docs/quant/ascend_kv_quant.md) for details.

## Limitations on 300I Duo

1. only support dtype=float16.
2. only support graph mode, please do not add --eager-mode.
