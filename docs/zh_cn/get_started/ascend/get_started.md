# 华为昇腾（Atlas 800T A2 & Atlas 300I Duo）

我们基于 LMDeploy 的 PytorchEngine，增加了华为昇腾设备的支持。所以，在华为昇腾上使用 LMDeploy 的方法与在英伟达 GPU 上使用 PytorchEngine 后端的方法几乎相同。在阅读本教程之前，请先阅读原版的[快速开始](../get_started.md)。

支持的模型列表在[这里](../../supported_models/supported_models.md#PyTorchEngine-华为昇腾平台).

> \[!IMPORTANT\]
> 我们已经在阿里云上提供了构建完成的鲲鹏CPU版本的镜像(从lmdeploy 0.7.1 + dlinfer 0.1.6开始)。
> 请使用下面的命令来拉取镜像:
> Atlas 800T A2:
> `docker pull crpi-4crprmm5baj1v8iv.cn-hangzhou.personal.cr.aliyuncs.com/lmdeploy_dlinfer/ascend:910b-latest`
> Atlas 300I Duo:
> `docker pull crpi-4crprmm5baj1v8iv.cn-hangzhou.personal.cr.aliyuncs.com/lmdeploy_dlinfer/ascend:310p-latest`
> 下述的dockerfile依然是可以执行的，您可以直接拉取镜像，也可以使用dockerfile来自己构建。

## 安装

我们强烈建议用户构建一个 Docker 镜像以简化环境设置。

克隆 lmdeploy 的源代码，Dockerfile 位于 docker 目录中。

```shell
git clone https://github.com/InternLM/lmdeploy.git
cd lmdeploy
```

### 环境准备

Docker 版本应不低于 18.09。并且需按照[官方指南](https://www.hiascend.com/document/detail/zh/mindx-dl/600/clusterscheduling/clusterschedulingig/clusterschedulingig/dlug_installation_012.html)安装 Ascend Docker Runtime。

> \[!CAUTION\]
> 如果在后续容器内出现`libascend_hal.so: cannot open shared object file`错误，说明Ascend Docker Runtime没有被正确安装。

#### Drivers，Firmware 和 CANN

目标机器需安装华为驱动程序和固件版本至少为 23.0.3，请参考
[CANN 驱动程序和固件安装](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/800alpha001/softwareinst/instg/instg_0005.html)
和[下载资源](https://www.hiascend.com/hardware/firmware-drivers/community?product=4&model=26&cann=8.0.RC2.beta1&driver=1.0.25.alpha)。

另外，`docker/Dockerfile_aarch64_ascend`没有提供CANN 安装包，用户需要自己从[昇腾资源下载中心](https://www.hiascend.com/developer/download/community/result?module=cann&cann=8.0.RC2.beta1&product=4&model=26)下载CANN(version 8.0.RC2.beta1)软件包。
并将`Ascend-cann-kernels-910b*.run`，`Ascend-cann-nnal_*.run`和`Ascend-cann-toolkit*.run` 放在 lmdeploy 源码根目录下。

#### 构建镜像

请在 lmdeploy源 代码根目录下执行以下镜像构建命令，CANN 相关的安装包也放在此目录下。

```bash
DOCKER_BUILDKIT=1 docker build -t lmdeploy-aarch64-ascend:latest \
    -f docker/Dockerfile_aarch64_ascend .
```

上述`Dockerfile_aarch64_ascend`适用于鲲鹏CPU. 如果是Intel CPU的机器，请尝试[这个dockerfile](https://github.com/InternLM/lmdeploy/issues/2745#issuecomment-2473285703) (未经过测试)

如果以下命令执行没有任何错误，这表明环境设置成功。

```bash
docker run -e ASCEND_VISIBLE_DEVICES=0 --rm --name lmdeploy -t lmdeploy-aarch64-ascend:latest lmdeploy check_env
```

关于在昇腾设备上运行`docker run`命令的详情，请参考这篇[文档](https://www.hiascend.com/document/detail/zh/mindx-dl/600/clusterscheduling/dockerruntimeug/dlruntime_ug_013.html)。

## 离线批处理

> \[!TIP\]
> 图模式已经支持了Atlas 800T A2。用户可以设定`eager_mode=False`来开启图模式，或者设定`eager_mode=True`来关闭图模式。(启动图模式需要事先source `/usr/local/Ascend/nnal/atb/set_env.sh`)

### LLM 推理

将`device_type="ascend"`加入`PytorchEngineConfig`的参数中。

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

### VLM 推理

将`device_type="ascend"`加入`PytorchEngineConfig`的参数中。

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

## 在线服务

> \[!TIP\]
> 图模式已经支持Atlas 800T A2。
> 在线服务时，图模式默认开启，用户可以添加`--eager-mode`来关闭图模式。(启动图模式需要事先source `/usr/local/Ascend/nnal/atb/set_env.sh`)

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
lmdeploy serve api_server --backend pytorch --device ascend --eager-mode OpenGVLab/InternVL2-2B
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
