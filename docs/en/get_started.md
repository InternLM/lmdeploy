# Get Started

LMDeploy offers functionalities such as model quantization, offline batch inference, online serving, etc. Each function can be completed with just a few simple lines of code or commands.

## Installation

Install lmdeploy with pip (python 3.8+) or [from source](./build.md)

```shell
pip install lmdeploy
```

The default prebuilt package is compiled on CUDA 11.8. However, if CUDA 12+ is required, you can install lmdeploy by:

```shell
export LMDEPLOY_VERSION=0.2.0
export PYTHON_VERSION=38
pip install https://github.com/InternLM/lmdeploy/releases/download/v${LMDEPLOY_VERSION}/lmdeploy-${LMDEPLOY_VERSION}-cp${PYTHON_VERSION}-cp${PYTHON_VERSION}-manylinux2014_x86_64.whl
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

LMDeploy offers various serving methods, choosing one that best meet your requirements.

- [Serving with openai compatible server](https://lmdeploy.readthedocs.io/en/latest/serving/restful_api.html)
- [Serving with docker](https://lmdeploy.readthedocs.io/en/latest/serving/restful_api.html#option-2-deploying-with-docker)
- [Serving with gradio](https://lmdeploy.readthedocs.io/en/latest/serving/gradio.html)

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
