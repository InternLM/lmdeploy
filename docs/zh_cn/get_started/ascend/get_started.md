# 华为昇腾（Atlas 800T A3 & Atlas 800T A2 & Atlas 300I Duo）

我们基于 LMDeploy 的 PytorchEngine，增加了华为昇腾设备的支持。所以，在华为昇腾上使用 LDMeploy 的方法与在英伟达 GPU 上使用 PytorchEngine 后端的方法几乎相同。在阅读本教程之前，请先阅读原版的[快速开始](../get_started.md)。

支持的模型列表在[这里](../../supported_models/supported_models.md#PyTorchEngine-华为昇腾平台).

> \[!IMPORTANT\]
> 我们已经在阿里云上提供了构建完成的鲲鹏CPU版本的镜像。
> 请使用下面的命令来拉取镜像:
> 
> Atlas 800T A3:
> `docker pull crpi-4crprmm5baj1v8iv.cn-hangzhou.personal.cr.aliyuncs.com/lmdeploy_dlinfer/ascend:a3-latest`
> （Atlas 800T A3目前只支持Qwen系列的算子模式下运行）
> 
> Atlas 800T A2:
> `docker pull crpi-4crprmm5baj1v8iv.cn-hangzhou.personal.cr.aliyuncs.com/lmdeploy_dlinfer/ascend:910b-latest`
> 
> Atlas 300I Duo:
> `docker pull crpi-4crprmm5baj1v8iv.cn-hangzhou.personal.cr.aliyuncs.com/lmdeploy_dlinfer/ascend:310p-latest`
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
lmdeploy serve api_server --backend pytorch --device ascend --eager-mode internlm/internlm2_5-7b-chat
```

也可以运行以下命令启动容器运行LLM模型服务。

```bash
docker exec -it --net=host crpi-4crprmm5baj1v8iv.cn-hangzhou.personal.cr.aliyuncs.com/lmdeploy_dlinfer/ascend:latest \
    bash -i -c "lmdeploy serve api_server --backend pytorch --device ascend --eager-mode internlm/internlm2_5-7b-chat"
```

### VLM 模型服务

将`--device ascend`加入到服务启动命令中。

```bash
lmdeploy serve api_server --backend pytorch --device ascend OpenGVLab/InternVL2-2B
```

也可以运行以下命令启动容器运行VLM模型服务。

```bash
docker exec -it --net=host crpi-4crprmm5baj1v8iv.cn-hangzhou.personal.cr.aliyuncs.com/lmdeploy_dlinfer/ascend:latest \
    bash -i -c "lmdeploy serve api_server --backend pytorch --device ascend --eager-mode OpenGVLab/InternVL2-2B"
```

## 使用命令行与LLM模型对话

将`--device ascend`加入到服务启动命令中。

```bash
lmdeploy chat internlm/internlm2_5-7b-chat --backend pytorch --device ascend --eager-mode
```

也可以运行以下命令使启动容器后开启lmdeploy聊天

```bash
docker exec -it crpi-4crprmm5baj1v8iv.cn-hangzhou.personal.cr.aliyuncs.com/lmdeploy_dlinfer/ascend:latest \
    bash -i -c "lmdeploy chat --backend pytorch --device ascend --eager-mode internlm/internlm2_5-7b-chat"
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
