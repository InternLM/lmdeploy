# Get Started with Huawei Ascend

We currently support running lmdeploy on **Atlas 800T A3, Atlas 800T A2 and Atlas 300I Duo**.
The usage of lmdeploy on a Huawei Ascend device is almost the same as its usage on CUDA with PytorchEngine in lmdeploy.
Please read the original [Get Started](../get_started.md) guide before reading this tutorial.

Here is the [supported model list](../../supported_models/supported_models.md#pytorchengine-on-other-platforms).

## Native A3 installation

The current A3 PyTorch backend is implemented inside LMDeploy and does not
require dlinfer. Use a separate Python environment without dlinfer installed.
Load the CANN and NNAL/ATB environment scripts for your installation first.
The following CANN 9.1.0 / Python 3.12 source-install profile matches
[`Dockerfile_ascend_a3`](../../../../docker/Dockerfile_ascend_a3):

```bash
python3.12 -m venv .venv-ascend
source .venv-ascend/bin/activate
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
A3_WHEEL_DIR=$(mktemp -d)
python -m pip download --no-deps --only-binary=:all: --dest "$A3_WHEEL_DIR" \
    --index-url https://download.pytorch.org/whl/cpu \
    -c requirements/ascend_a3_constraints.txt torch torchvision
python -m pip download --no-deps --only-binary=:all: --dest "$A3_WHEEL_DIR" \
    --index-url https://mirrors.huaweicloud.com/ascend/repos/pypi \
    -r requirements/ascend_kernels.txt
LMDEPLOY_TARGET_DEVICE=ascend python -m pip install --index-url https://pypi.org/simple \
    --find-links "$A3_WHEEL_DIR" -c requirements/ascend_a3_constraints.txt \
    -r requirements/ascend_kernels.txt . pyarrow
python -m pip check
python tools/ascend/check_runtime.py --require-kernels
```

Run these commands from the repository root. CANN, torch_npu and Triton-Ascend
remain required. The GDN prefill/L2Norm kernels are bundled with source provenance
and their MIT license; no external kernel repository checkout is needed. The
image uses an isolated application venv with Torch 2.10.0+cpu, torch_npu
2.10.0.post4, torchvision 0.25.0+cpu and Triton-Ascend 3.2.2.post2. Keep `device_type="ascend"`
or `--device ascend` in inference commands. Graph mode uses `torch.npu.NPUGraph`;
use `eager_mode=True` or `--eager-mode` to disable it.

The prebuilt images below describe older dlinfer-based releases. The native
migration targets A3; A2 and Atlas 300I Duo have not been validated with this
backend. Build the current A3 checkout with `docker/Dockerfile_ascend_a3` when
using containers. Its CANN/PyTorch recipe must match the host driver and runtime.

### GDN/MTP state correctness and compiler updates

The circular GDN state path uses `num_stages=1`; other recurrent paths retain
`num_stages=3`. AscendNPU-IR 1.2.0 build `042c923d` can corrupt intermediate
FP32 state writes with automatic multibuffering, even when the current output
is correct. MTP later reads those saved states, causing greedy target logits
and generated tokens to differ across runs. Single buffering protects this
path on the affected compiler and remains the default on newer compilers:
correct multibuffering was not faster for every tested shape.

Upstream fix [`05f2adcf`](https://github.com/Ascend/AscendNPU-IR/commit/05f2adcf3db366377f7c5e6f5bb9bd3c61c36b3e)
is included in AscendNPU-IR 2.0.0 (`efb71415`), shipped in the official
Triton-Ascend 3.2.2.post2 wheel. For an existing compatible native A3 runtime,
the optional update profile is:

```bash
python -m pip install --no-deps -r requirements/ascend_triton_npuir2.txt \
    --extra-index-url https://mirrors.huaweicloud.com/ascend/repos/pypi
```

Apply this in an independent environment with its own package files. New
installations already select this compiler through `requirements/ascend_kernels.txt`.
The component-update checks used Python 3.12.13, Torch 2.10.0+cpu,
torch_npu 2.10.0.post4 and CANN 9.1.0. The Dockerfile now uses the official
`9.1.0-a3-openeuler24.03-py3.12` image and the matching pinned dependency profile.
The [official preferred NPUIR pairing](https://ascendnpu-ir.gitcode.com/en/introduction/quick_start/version_compatibility.html)
is NPUIR 2.0.0 with CANN 9.2.0; here the tested compiler update is used with the
existing CANN 9.1.0 runtime. This does not upgrade the host driver or firmware.

Triton prefers its bundled compiler over PATH. Check the executable actually
selected after the update, rather than only `triton.__version__`:

```python
import subprocess
from triton.backends.ascend.utils import _get_npucompiler_path

compiler, compiler_env = _get_npucompiler_path()
print(compiler)
subprocess.run([compiler, '--version'], env=compiler_env, check=True)
```

Run the state regression on A3 with:

```bash
python -m pytest -q --run-ascend-hardware \
    tests/pytorch/ascend/hardware/test_gdn_state_writeback.py
```

The test checks every saved state against a CPU recurrence for Q=1/2/4 in eager
and graph modes, restoring the initial state on every replay. The
[investigation report](https://github.com/silencelamb/lmdeploy/blob/docs/ascend-a3-validation-20261114/docs/zh_cn/advance/ascend_a3_validation/mtp_ray_investigation_20261116.md)
records the compiler IR, fixed versions, MTP comparisons, shape-dependent
performance and remaining coverage. Operator and target-backbone timings do
not establish end-to-end serving throughput.

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
