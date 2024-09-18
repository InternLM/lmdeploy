# Get Started with Huawei Ascend (Atlas 800T A2)

The usage of lmdeploy on a Huawei Ascend device is almost the same as its usage on CUDA with PytorchEngine in lmdeploy.
Please read the original [Get Started](../get_started.md) guide before reading this tutorial.

## Installation

We highly recommend that users build a Docker image for streamlined environment setup.

Git clone the source code of lmdeploy and the Dockerfile locates in the `docker` directory:

```shell
git clone https://github.com/InternLM/lmdeploy.git
cd lmdeploy
```

### Environment Preparation

The Docker version is supposed to be no less than `18.03`. And `Ascend Docker Runtime` should be installed by following [the official guide](https://www.hiascend.com/document/detail/zh/mindx-dl/60rc2/clusterscheduling/clusterschedulingig/.clusterschedulingig/dlug_installation_012.html).

#### Ascend Drivers, Firmware and CANN

The target machine needs to install the Huawei driver and firmware version 23.0.3, refer to
[CANN Driver and Firmware Installation](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/80RC1alpha003/softwareinst/instg/instg_0019.html)
and [download resources](https://www.hiascend.com/hardware/firmware-drivers/community?product=4&model=26&cann=8.0.RC3.alpha001&driver=1.0.0.2.alpha).

And the CANN (version 8.0.RC3.alpha001) software packages should also be downloaded from [Ascend Resource Download Center](https://www.hiascend.com/developer/download/community/result?module=cann&cann=8.0.RC3.alpha001) themselves. Make sure to place the `Ascend-cann-kernels-910b*.run` and `Ascend-cann-toolkit*-aarch64.run` under the root directory of lmdeploy source code

#### Build Docker Image

Run the following command in the root directory of lmdeploy to build the image:

```bash
DOCKER_BUILDKIT=1 docker build -t lmdeploy-aarch64-ascend:latest \
    -f docker/Dockerfile_aarch64_ascend .
```

If the following command executes without any errors, it indicates that the environment setup is successful.

```bash
docker run -e ASCEND_VISIBLE_DEVICES=0 --rm --name lmdeploy -t lmdeploy-aarch64-ascend:latest lmdeploy check_env
```

For more information about running the Docker client on Ascend devices, please refer to the [guide](https://www.hiascend.com/document/detail/zh/mindx-dl/60rc1/clusterscheduling/dockerruntimeug/dlruntime_ug_013.html)

## Offline batch inference

### LLM inference

Set `device_type="ascend"` in the `PytorchEngineConfig`:

```python
from lmdeploy import pipeline
from lmdeploy import PytorchEngineConfig
if __name__ == "__main__":
    pipe = pipeline("internlm/internlm2_5-7b-chat",
                    backend_config = PytorchEngineConfig(tp=1, device_type="ascend"))
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

### Serve a VLM model

Add `--device ascend` in the serve command

```bash
lmdeploy serve api_server --backend pytorch --device ascend OpenGVLab/InternVL2-2B
```

## Inference with Command line Interface

Add `--device ascend` in the serve command.

```bash
lmdeploy chat internlm/internlm2_5-7b-chat --backend pytorch --device ascend
```

Run the following commands to launch lmdeploy chatting after starting container:

```bash
docker exec -it lmdeploy_ascend_demo \
    bash -i -c "lmdeploy chat --backend pytorch --device ascend internlm/internlm2_5-7b-chat"
```
