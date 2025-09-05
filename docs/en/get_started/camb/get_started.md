# Cambricon

The usage of lmdeploy on a Cambricon device is almost the same as its usage on CUDA with PytorchEngine in lmdeploy.
Please read the original [Get Started](../get_started.md) guide before reading this tutorial.

Here is the [supported model list](../../supported_models/supported_models.md#PyTorchEngine-on-Other-Platforms).

> \[!IMPORTANT\]
> We have uploaded a docker image to aliyun.
> Please try to pull the image by following command:
>
> `docker pull crpi-4crprmm5baj1v8iv.cn-hangzhou.personal.cr.aliyuncs.com/lmdeploy_dlinfer/camb:latest`

## Offline batch inference

### LLM inference

Set `device_type="camb"` in the `PytorchEngineConfig`:

```python
from lmdeploy import pipeline
from lmdeploy import PytorchEngineConfig
pipe = pipeline("internlm/internlm2_5-7b-chat",
        backend_config=PytorchEngineConfig(tp=1, device_type="camb"))
question = ["Shanghai is", "Please introduce China", "How are you?"]
response = pipe(question)
print(response)
```

### VLM inference

Set `device_type="camb"` in the `PytorchEngineConfig`:

```python
from lmdeploy import pipeline, PytorchEngineConfig
from lmdeploy.vl import load_image
pipe = pipeline('OpenGVLab/InternVL2-2B',
        backend_config=PytorchEngineConfig(tp=1, device_type='camb'))
image = load_image('https://raw.githubusercontent.com/open-mmlab/mmdeploy/main/tests/data/tiger.jpeg')
response = pipe(('describe this image', image))
print(response)
```

## Online serving

### Serve a LLM model

Add `--device camb` in the serve command.

```bash
lmdeploy serve api_server --backend pytorch --device camb internlm/internlm2_5-7b-chat
```

Run the following commands to launch docker container for lmdeploy LLM serving:

```bash
docker run -it --net=host crpi-4crprmm5baj1v8iv.cn-hangzhou.personal.cr.aliyuncs.com/lmdeploy_dlinfer/camb:latest \
    bash -i -c "lmdeploy serve api_server --backend pytorch --device camb internlm/internlm2_5-7b-chat"
```

### Serve a VLM model

Add `--device camb` in the serve command

```bash
lmdeploy serve api_server --backend pytorch --device camb OpenGVLab/InternVL2-2B
```

Run the following commands to launch docker container for lmdeploy VLM serving:

```bash
docker run -it --net=host crpi-4crprmm5baj1v8iv.cn-hangzhou.personal.cr.aliyuncs.com/lmdeploy_dlinfer/camb:latest \
    bash -i -c "lmdeploy serve api_server --backend pytorch --device camb OpenGVLab/InternVL2-2B"
```

## Inference with Command line Interface

Add `--device camb` in the serve command.

```bash
lmdeploy chat internlm/internlm2_5-7b-chat --backend pytorch --device camb
```

Run the following commands to launch lmdeploy chatting after starting container:

```bash
docker run -it crpi-4crprmm5baj1v8iv.cn-hangzhou.personal.cr.aliyuncs.com/lmdeploy_dlinfer/camb:latest \
    bash -i -c "lmdeploy chat --backend pytorch --device camb internlm/internlm2_5-7b-chat"
```
