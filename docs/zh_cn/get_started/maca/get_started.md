# 沐曦C500

我们基于 LMDeploy 的 PytorchEngine，增加了沐曦C500设备的支持。所以，在沐曦上使用 LMDeploy 的方法与在英伟达 GPU 上使用 PytorchEngine 后端的方法几乎相同。在阅读本教程之前，请先阅读原版的[快速开始](../get_started.md)。

支持的模型列表在[这里](../../supported_models/supported_models.md#PyTorchEngine-其他平台).

> \[!IMPORTANT\]
> 我们已经在阿里云上提供了构建完成的沐曦的镜像。
> 请使用下面的命令来拉取镜像:
> `docker pull crpi-4crprmm5baj1v8iv.cn-hangzhou.personal.cr.aliyuncs.com/lmdeploy_dlinfer/maca:latest`

## 离线批处理

### LLM 推理

将`device_type="maca"`加入`PytorchEngineConfig`的参数中。

```python
from lmdeploy import pipeline
from lmdeploy import PytorchEngineConfig
pipe = pipeline("internlm/internlm2_5-7b-chat",
             backend_config=PytorchEngineConfig(tp=1, device_type="maca"))
question = ["Shanghai is", "Please introduce China", "How are you?"]
response = pipe(question)
print(response)
```

### VLM 推理

将`device_type="maca"`加入`PytorchEngineConfig`的参数中。

```python
from lmdeploy import pipeline, PytorchEngineConfig
from lmdeploy.vl import load_image
pipe = pipeline('OpenGVLab/InternVL2-2B',
     backend_config=PytorchEngineConfig(tp=1, device_type='maca'))
image = load_image('https://raw.githubusercontent.com/open-mmlab/mmdeploy/main/tests/data/tiger.jpeg')
response = pipe(('describe this image', image))
print(response)
```

## 在线服务

### LLM 模型服务

将`--device maca`加入到服务启动命令中。

```bash
lmdeploy serve api_server --backend pytorch --device maca internlm/internlm2_5-7b-chat
```

也可以运行以下命令启动容器运行LLM模型服务。

```bash
docker run -it --net=host crpi-4crprmm5baj1v8iv.cn-hangzhou.personal.cr.aliyuncs.com/lmdeploy_dlinfer/maca:latest \
    bash -i -c "lmdeploy serve api_server --backend pytorch --device maca internlm/internlm2_5-7b-chat"
```

### VLM 模型服务

将`--device maca`加入到服务启动命令中。

```bash
lmdeploy serve api_server --backend pytorch --device maca OpenGVLab/InternVL2-2B
```

也可以运行以下命令启动容器运行VLM模型服务。

```bash
docker run -it --net=host crpi-4crprmm5baj1v8iv.cn-hangzhou.personal.cr.aliyuncs.com/lmdeploy_dlinfer/maca:latest \
    bash -i -c "lmdeploy serve api_server --backend pytorch --device maca OpenGVLab/InternVL2-2B"
```

## 使用命令行与LLM模型对话

将`--device maca`加入到服务启动命令中。

```bash
lmdeploy chat internlm/internlm2_5-7b-chat --backend pytorch --device maca
```

也可以运行以下命令使启动容器后开启lmdeploy聊天

```bash
docker run -it crpi-4crprmm5baj1v8iv.cn-hangzhou.personal.cr.aliyuncs.com/lmdeploy_dlinfer/maca:latest \
    bash -i -c "lmdeploy chat --backend pytorch --device maca internlm/internlm2_5-7b-chat"
```
