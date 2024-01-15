# 快速上手

LMDeploy提供了快速安装、模型量化、离线批处理、在线推理服务等功能。每个功能只需简单的几行代码或者命令就可以完成。

## 安装

使用 pip ( python 3.8+) 安装 LMDeploy，或者[源码安装](./build.md)

```shell
pip install lmdeploy
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

LMDeploy `api_server` 支持把模型一键封装为服务，对外提供的 RESTful API 兼容 openai 的接口。以下为服务启动的示例：

```shell
lmdeploy serve api_server internlm/internlm-chat-7b --server-port 8080
```

服务启动后，你可以在终端通过`api_client`与server进行对话：

```shell
lmdeploy serve api_client http://0.0.0.0:8080
```

除了`api_client`，你还可以通过 Swagger UI `http://0.0.0.0:8080` 在线阅读和试用 `api_server` 的各接口，也可直接查阅[文档](serving/restful_api.md)，了解各接口的定义和使用方法。

## 模型量化

- [INT4 权重量化](quantization/w4a16.md)
- [K/V 量化](quantization/kv_int8.md)
- [W8A8 量化](quantization/w8a8.md)

## 好用的工具

LMDeploy CLI 提供了如下便捷的工具，方便用户快速体验模型对话效果

### 控制台交互式对话

```shell
lmdeploy chat turbomind internlm/internlm-chat-7b
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
