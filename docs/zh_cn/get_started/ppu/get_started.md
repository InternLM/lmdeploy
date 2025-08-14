# 在阿里平头哥上快速开始

我们基于 LMDeploy 的 PytorchEngine，增加了平头哥设备的支持。所以，在平头哥上使用 LMDeploy 的方法与在英伟达 GPU 上使用 PytorchEngine 后端的方法几乎相同。在阅读本教程之前，请先阅读原版的[快速开始](../get_started.md)。

## 安装

安装请参考 [dlinfer 安装方法](https://github.com/DeepLink-org/dlinfer#%E5%AE%89%E8%A3%85%E6%96%B9%E6%B3%95)。

## 离线批处理

> \[!TIP\]
> 图模式已支持。用户可以设定`eager_mode=False`来开启图模式，或者设定`eager_mode=True`来关闭图模式。

### LLM 推理

将`device_type="ppu"`加入`PytorchEngineConfig`的参数中。

```python
from lmdeploy import pipeline
from lmdeploy import PytorchEngineConfig
if __name__ == "__main__":
    pipe = pipeline("internlm/internlm2_5-7b-chat",
                    backend_config=PytorchEngineConfig(tp=1, device_type="ppu", eager_mode=True))
    question = ["Shanghai is", "Please introduce China", "How are you?"]
    response = pipe(question)
    print(response)
```

### VLM 推理

将`device_type="ppu"`加入`PytorchEngineConfig`的参数中。

```python
from lmdeploy import pipeline, PytorchEngineConfig
from lmdeploy.vl import load_image
if __name__ == "__main__":
    pipe = pipeline('OpenGVLab/InternVL2-2B',
                    backend_config=PytorchEngineConfig(tp=1, device_type='ppu', eager_mode=True))
    image = load_image('https://raw.githubusercontent.com/open-mmlab/mmdeploy/main/tests/data/tiger.jpeg')
    response = pipe(('describe this image', image))
    print(response)
```

## 在线服务

> \[!TIP\]
> 图模式已支持。
> 在线服务时，图模式默认开启，用户可以添加`--eager-mode`来关闭图模式。

### LLM 模型服务

将`--device ppu`加入到服务启动命令中。

```bash
lmdeploy serve api_server --backend pytorch --device ppu --eager-mode internlm/internlm2_5-7b-chat
```

### VLM 模型服务

将`--device ppu`加入到服务启动命令中。

```bash
lmdeploy serve api_server --backend pytorch --device ppu --eager-mode OpenGVLab/InternVL2-2B
```

## 使用命令行与LLM模型对话

将`--device ppu`加入到服务启动命令中。

```bash
lmdeploy chat internlm/internlm2_5-7b-chat --backend pytorch --device ppu --eager-mode
```
