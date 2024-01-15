# Get Started

LMDeploy offers functionalities such as model quantization, offline batch inference, online serving, etc. Each function can be completed with just a few simple lines of code or commands.

## Installation

Install lmdeploy with pip (python 3.8+) or [from source](./build.md)

```shell
pip install lmdeploy
```

## Offline batch inference

```python
import lmdeploy
pipe = lmdeploy.pipeline("internlm/internlm-chat-7b")
response = pipe(["Hi, pls intro yourself", "Shanghai is"])
print(response)
```

For more information on inference pipeline parameters, please refer to [here](./inference/pipeline.md).

## Serving

LMDeploy's `api_server` enables models to be easily packed into services with a single command. The provided RESTful APIs are compatible with OpenAI's interfaces. Below are an example of service startup:

```shell
lmdeploy serve api_server internlm/internlm-chat-7b
```

The default port of `api_server` is `23333`. After the server is launched, you can communicate with server on terminal through `api_client`:

```shell
lmdeploy serve api_client http://0.0.0.0:23333
```

Besides the `api_client`, you can overview and try out `api_server` APIs online by swagger UI `http://0.0.0.0:23333`. And you can also read the API specification from [here](serving/restful_api.md).

## Quantization

LMDeploy provides the following quantization methods. Please visit the following links for the detailed guide

- [4bit weight-only quantization](quantization/w4a16.md)
- [k/v quantization](quantization/kv_int8.md)
- [w8a8 quantization](quantization/w8a8.md)

## Useful Tools

LMDeploy CLI offers the following utilities, helping users experience LLM features conveniently

### Inference with Command line Interface

```shell
lmdeploy chat turbomind internlm/internlm-chat-7b
```

### Serving with Web UI

LMDeploy adopts gradio to develop the online demo.

```shell
# install dependencies
pip install lmdeploy[serve]
# launch gradio server
lmdeploy serve gradio internlm/internlm-chat-7b
```

![](https://github.com/InternLM/lmdeploy/assets/67539920/08d1e6f2-3767-44d5-8654-c85767cec2ab)
