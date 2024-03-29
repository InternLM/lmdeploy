# 快速上手

LMDeploy提供了快速安装、模型量化、离线批处理、在线推理服务等功能。每个功能只需简单的几行代码或者命令就可以完成。

## 安装

使用 pip (python 3.8+) 安装 LMDeploy，或者[源码安装](./build.md)

```shell
pip install lmdeploy
```

LMDeploy的预编译包默认是基于 CUDA 12.1 编译的。如果需要在 CUDA 11+ 下安装 LMDeploy，请执行以下命令：

```shell
export LMDEPLOY_VERSION=0.2.0
export PYTHON_VERSION=38
pip install https://github.com/InternLM/lmdeploy/releases/download/v${LMDEPLOY_VERSION}/lmdeploy-${LMDEPLOY_VERSION}+cu118-cp${PYTHON_VERSION}-cp${PYTHON_VERSION}-manylinux2014_x86_64.whl
```

## 离线批处理

```python
import lmdeploy
pipe = lmdeploy.pipeline("internlm/internlm-chat-7b")
response = pipe(["Hi, pls intro yourself", "Shanghai is"])
print(response)
```

有关 pipeline 的详细使用说明，请参考[这里](./inference/pipeline.md)

## 推理服务

LMDeploy 提供了多种部署模型推理服务的方式，总有一款适合你。

- [部署类 openai 的服务](https://lmdeploy.readthedocs.io/zh-cn/latest//serving/api_server.html)
- [通过 docker 部署服务](https://lmdeploy.readthedocs.io/zh-cn/latest/serving/api_server.html#docker)
- [部署 gradio 服务](https://lmdeploy.readthedocs.io/zh-cn/latest/serving/gradio.html)

## 模型量化

- [INT4 权重量化](quantization/w4a16.md)
- [K/V 量化](quantization/kv_int8.md)
- [W8A8 量化](quantization/w8a8.md)

## 好用的工具

LMDeploy CLI 提供了如下便捷的工具，方便用户快速体验模型对话效果

### 控制台交互式对话

```shell
lmdeploy chat internlm/internlm-chat-7b
```

### WebUI 交互式对话

LMDeploy 使用 gradio 开发了在线对话 demo。

```shell
# 安装依赖
pip install lmdeploy[serve]
# 启动
lmdeploy serve gradio internlm/internlm-chat-7b
```

![](https://github.com/InternLM/lmdeploy/assets/67539920/08d1e6f2-3767-44d5-8654-c85767cec2ab)
