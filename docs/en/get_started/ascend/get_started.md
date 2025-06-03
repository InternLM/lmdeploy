# Get Started with Huawei Ascend (Atlas 800T A2 & Atlas 300I Duo)

The usage of lmdeploy on a Huawei Ascend device is almost the same as its usage on CUDA with PytorchEngine in lmdeploy.
Please read the original [Get Started](../get_started.md) guide before reading this tutorial.

Here is the [supported model list](../../supported_models/supported_models.md#PyTorchEngine-on-Huawei-Ascend-Platform).

> \[!IMPORTANT\]
> We have uploaded a docker image with KUNPENG CPU to aliyun(from lmdeploy 0.7.1 + dlinfer 0.1.6).
> Please try to pull the image by following command:
> Atlas 800T A2:
> `docker pull crpi-4crprmm5baj1v8iv.cn-hangzhou.personal.cr.aliyuncs.com/lmdeploy_dlinfer/ascend:910b-latest`
> 300I Duo:
> `docker pull crpi-4crprmm5baj1v8iv.cn-hangzhou.personal.cr.aliyuncs.com/lmdeploy_dlinfer/ascend:310p-latest`
> The dockerfile described below still works, you can try
> both pulling image and build your own image by dockerfile.

## Installation

We highly recommend that users build a Docker image for streamlined environment setup.

Git clone the source code of lmdeploy and the Dockerfile locates in the `docker` directory:

```shell
git clone https://github.com/InternLM/lmdeploy.git
cd lmdeploy
```

### Environment Preparation

The Docker version is supposed to be no less than `18.09`. And `Ascend Docker Runtime` should be installed by following [the official guide](https://www.hiascend.com/document/detail/zh/mindx-dl/60rc2/clusterscheduling/clusterschedulingig/.clusterschedulingig/dlug_installation_012.html).

> \[!CAUTION\]
> If error message `libascend_hal.so: cannot open shared object file` shows, that means **Ascend Docker Runtime** is not installed correctly!

#### Ascend Drivers, Firmware and CANN

The target machine needs to install the Huawei driver and firmware version not lower than 23.0.3, refer to
[CANN Driver and Firmware Installation](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/800alpha001/softwareinst/instg/instg_0005.html)
and [download resources](https://www.hiascend.com/hardware/firmware-drivers/community?product=4&model=26&cann=8.0.RC2.beta1&driver=1.0.25.alpha).

And the CANN (version 8.0.RC2.beta1) software packages should also be downloaded from [Ascend Resource Download Center](https://www.hiascend.com/developer/download/community/result?module=cann&cann=8.0.RC2.beta1&product=4&model=26) themselves. Make sure to place the `Ascend-cann-kernels-910b*.run`, `Ascend-cann-nnal_*.run` and `Ascend-cann-toolkit*-aarch64.run` under the root directory of lmdeploy source code

#### Build Docker Image

Run the following command in the root directory of lmdeploy to build the image:

```bash
DOCKER_BUILDKIT=1 docker build -t lmdeploy-aarch64-ascend:latest \
    -f docker/Dockerfile_aarch64_ascend .
```

The `Dockerfile_aarch64_ascend` is tested on Kunpeng CPU. For intel CPU, please try [this dockerfile](https://github.com/InternLM/lmdeploy/issues/2745#issuecomment-2473285703) (which is not fully tested)

If the following command executes without any errors, it indicates that the environment setup is successful.

```bash
docker run -e ASCEND_VISIBLE_DEVICES=0 --rm --name lmdeploy -t lmdeploy-aarch64-ascend:latest lmdeploy check_env
```

For more information about running the Docker client on Ascend devices, please refer to the [guide](https://www.hiascend.com/document/detail/zh/mindx-dl/60rc1/clusterscheduling/dockerruntimeug/dlruntime_ug_013.html)

## Offline batch inference

> \[!TIP\]
> Graph mode has been supported on Atlas 800T A2.
> Users can set `eager_mode=False` to enable graph mode, or, set `eager_mode=True` to disable graph mode.
> (Please source `/usr/local/Ascend/nnal/atb/set_env.sh` before enabling graph mode)

### LLM inference

Set `device_type="ascend"` in the `PytorchEngineConfig`:

```python
from lmdeploy import pipeline
from lmdeploy import PytorchEngineConfig
if __name__ == "__main__":
    pipe = pipeline("internlm/internlm2_5-7b-chat",
                    backend_config=PytorchEngineConfig(tp=1, device_type="ascend", eager_mode=True))
    question = ["Shanghai is", "Please introduce China", "How are you?"]
    response = pipe(question)
    print(response)
```

### VLM inference

Set `device_type="ascend"` in the `PytorchEngineConfig`:

```python
from lmdeploy import pipeline, PytorchEngineConfig
from lmdeploy.vl import load_image
if __name__ == "__main__":
    pipe = pipeline('OpenGVLab/InternVL2-2B',
                    backend_config=PytorchEngineConfig(tp=1, device_type='ascend', eager_mode=True))
    image = load_image('https://raw.githubusercontent.com/open-mmlab/mmdeploy/main/tests/data/tiger.jpeg')
    response = pipe(('describe this image', image))
    print(response)
```

## Online serving

> \[!TIP\]
> Graph mode has been supported on Atlas 800T A2.
> Graph mode is default enabled in online serving. Users can add `--eager-mode` to disable graph mode.
> (Please source `/usr/local/Ascend/nnal/atb/set_env.sh` before enabling graph mode)

### Serve a LLM model

Add `--device ascend` in the serve command.

```bash
lmdeploy serve api_server --backend pytorch --device ascend --eager-mode internlm/internlm2_5-7b-chat
```

Run the following commands to launch docker container for lmdeploy LLM serving:

```bash
docker exec -it --net=host crpi-4crprmm5baj1v8iv.cn-hangzhou.personal.cr.aliyuncs.com/lmdeploy_dlinfer/ascend:latest \
    bash -i -c "lmdeploy serve api_server --backend pytorch --device ascend --eager-mode internlm/internlm2_5-7b-chat"
```

### Serve a VLM model

Add `--device ascend` in the serve command

```bash
lmdeploy serve api_server --backend pytorch --device ascend --eager-mode OpenGVLab/InternVL2-2B
```

Run the following commands to launch docker container for lmdeploy VLM serving:

```bash
docker exec -it --net=host crpi-4crprmm5baj1v8iv.cn-hangzhou.personal.cr.aliyuncs.com/lmdeploy_dlinfer/ascend:latest \
    bash -i -c "lmdeploy serve api_server --backend pytorch --device ascend --eager-mode OpenGVLab/InternVL2-2B"
```

## Inference with Command line Interface

Add `--device ascend` in the serve command.

```bash
lmdeploy chat internlm/internlm2_5-7b-chat --backend pytorch --device ascend --eager-mode
```

Run the following commands to launch lmdeploy chatting after starting container:

```bash
docker exec -it crpi-4crprmm5baj1v8iv.cn-hangzhou.personal.cr.aliyuncs.com/lmdeploy_dlinfer/ascend:latest \
    bash -i -c "lmdeploy chat --backend pytorch --device ascend --eager-mode internlm/internlm2_5-7b-chat"
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
