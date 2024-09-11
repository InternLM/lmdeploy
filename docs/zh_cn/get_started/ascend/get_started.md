# 华为昇腾(Atlas 800T A2）

我们采用了LMDeploy中的PytorchEngine后端支持了华为昇腾设备，
所以在华为昇腾上使用lmdeploy的方法与在英伟达GPU上使用PytorchEngine后端的使用方法几乎相同。
在阅读本教程之前，请先阅读原版的[快速开始](../get_started.md)。

## 安装

### 环境准备

#### Drivers和Firmware

Host需要安装华为驱动程序和固件版本23.0.3，请参考
[CANN 驱动程序和固件安装](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/80RC1alpha003/softwareinst/instg/instg_0019.html)
和[下载资源](https://www.hiascend.com/hardware/firmware-drivers/community?product=4&model=26&cann=8.0.RC3.alpha001&driver=1.0.0.2.alpha)。

#### CANN

`docker/Dockerfile_aarch64_ascend`没有提供CANN 安装包，用户需要自己从[昇腾资源下载中心](https://www.hiascend.com/developer/download/community/result?module=cann&cann=8.0.RC3.alpha001)下载CANN(8.0.RC3.alpha001)软件包。
并将Ascend-cann-kernels-910b\*.run 和 Ascend-cann-toolkit\*-aarch64.run 放在执行`docker build`命令的目录下。

#### Docker

构建aarch64_ascend镜像需要Docker>=18.03

#### 构建镜像的命令

请在lmdeploy源代码根目录下执行以下镜像构建命令，CANN相关的安装包也放在此目录下。

```bash
DOCKER_BUILDKIT=1 docker build -t lmdeploy-aarch64-ascend:v0.1 \
    -f docker/Dockerfile_aarch64_ascend .
```

这个镜像将使用`pip install --no-build-isolation -e .`命令将lmdeploy安装到/workspace/lmdeploy目录。

#### 镜像的使用

关于镜像的使用方式，请参考这篇[文档](https://www.hiascend.com/document/detail/zh/mindx-dl/60rc1/clusterscheduling/dockerruntimeug/dlruntime_ug_013.html)。
并且在使用镜像前安装Ascend Docker Runtime。
以下是在安装了 Ascend Docker Runtime 的情况下，启动用于华为昇腾设备的容器的示例：

```bash
docker run -e ASCEND_VISIBLE_DEVICES=0 --net host -td --entrypoint bash --name lmdeploy_ascend_demo \
    lmdeploy-aarch64-ascend:v0.1  # docker_image_sha_or_name
```

#### 使用Pip安装

如果您已经安装了lmdeploy并且所有华为环境都已准备好，您可以运行以下命令使lmdeploy能够在华为昇腾设备上运行。(如果使用Docker镜像则不需要)

```bash
pip install dlinfer-ascend
```

## 离线批处理

### LLM 推理

将`device_type="ascend"`加入`PytorchEngineConfig`的参数中。

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

### VLM 推理

将`device_type="ascend"`加入`PytorchEngineConfig`的参数中。

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

## 在线服务

### LLM 模型服务

将`--device ascend`加入到服务启动命令中。

```bash
lmdeploy serve api_server --backend pytorch --device ascend internlm/internlm2_5-7b-chat
```

### VLM 模型服务

将`--device ascend`加入到服务启动命令中。

```bash
lmdeploy serve api_server --backend pytorch --device ascend OpenGVLab/InternVL2-2B
```

## 使用命令行与LLM模型对话

将`--device ascend`加入到服务启动命令中。

```bash
lmdeploy chat internlm/internlm2_5-7b-chat --backend pytorch --device ascend
```

也可以运行以下命令使启动容器后开启lmdeploy聊天

```bash
docker exec -it lmdeploy_ascend_demo \
    bash -i -c "lmdeploy chat --backend pytorch --device ascend internlm/internlm2_5-7b-chat"
```
