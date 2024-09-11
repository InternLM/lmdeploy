# Get Started with Huawei Ascend (Atlas 800T A2）

The usage of lmdeploy on a Huawei Ascend device is almost the same as its usage on CUDA with PytorchEngine in lmdeploy.
Please read the original [Get Started](../get_started.md) guide before reading this tutorial.

## Installation

### Environment Preparation

#### Drivers and Firmware

The host machine needs to install the Huawei driver and firmware version 23.0.3, refer to
[CANN Driver and Firmware Installation](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/80RC1alpha003/softwareinst/instg/instg_0019.html)
and [download resources](https://www.hiascend.com/hardware/firmware-drivers/community?product=4&model=26&cann=8.0.RC3.alpha001&driver=1.0.0.2.alpha).

#### CANN

File `docker/Dockerfile_aarch64_ascend` does not provide Ascend CANN installation package, users need to download the CANN (version 8.0.RC3.alpha001) software packages from [Ascend Resource Download Center](https://www.hiascend.com/developer/download/community/result?module=cann&cann=8.0.RC3.alpha001) themselves. And place the Ascend-cann-kernels-910b\*.run and Ascend-cann-toolkit\*-aarch64.run under the directory where the docker build command is executed.

#### Docker

Building the aarch64_ascend image requires Docker >= 18.03

#### Reference Command for Building the Image

The following reference command for building the image is based on the lmdeploy source code root directory as the current directory, and the CANN-related installation packages are also placed under this directory.

```bash
DOCKER_BUILDKIT=1 docker build -t lmdeploy-aarch64-ascend:v0.1 \
    -f docker/Dockerfile_aarch64_ascend .
```

This image will install lmdeploy to `/workspace/lmdeploy` directory using `pip install --no-build-isolation -e .` command.

#### Using the Image

You can refer to the [documentation](https://www.hiascend.com/document/detail/zh/mindx-dl/60rc1/clusterscheduling/dockerruntimeug/dlruntime_ug_013.html)
for usage. It is recommended to install Ascend Docker Runtime.
Here is an example of starting container for Huawei Ascend device with Ascend Docker Runtime installed:

```bash
docker run -e ASCEND_VISIBLE_DEVICES=0 --net host -td --entrypoint bash --name lmdeploy_ascend_demo \
    lmdeploy-aarch64-ascend:v0.1  # docker_image_sha_or_name
```

#### Pip install

If you have lmdeploy installed and all Huawei environments are ready, you can run the following command to enable lmdeploy to run on Huawei Ascend devices. (Not necessary if you use the Docker image.)

```bash
pip install dlinfer-ascend
```

## Offline batch inference

### LLM inference

Set `device_type="ascend"`  in the `PytorchEngineConfig`:

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
