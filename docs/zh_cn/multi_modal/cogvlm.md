# cogvlm

## 简介

CogVLM 是一个强大的开源视觉语言模型（VLM）. LMDeploy 已在PyTorch后端支持 CogVLM-17B 模型 [THUDM/cogvlm-chat-hf](https://huggingface.co/THUDM/cogvlm-chat-hf) 和 CogVLM2-19B 模型如[THUDM/cogvlm2-llama3-chat-19B](https://huggingface.co/THUDM/cogvlm2-llama3-chat-19B)

## 快速开始

请参考[安装文档](../get_started/installation.md)安装 LMDeploy

### 准备

当使用LMDeploy部署 **CogVLM** 模型时，需要下载模型至本地目录。由于 **CogVLM** 模型使用外部Tokenizer，因而需要将相关文件下载至模型目录。然而对于**CogVLM2**模型，则可跳过此步骤。

以 **CogVLM** 模型 `cogvlm-chat-hf` 为例，可执行如下脚本下载模型：

```shell
huggingface-cli download THUDM/cogvlm-chat-hf --local-dir ./cogvlm-chat-hf --local-dir-use-symlinks False
huggingface-cli download lmsys/vicuna-7b-v1.5 special_tokens_map.json tokenizer.model tokenizer_config.json --local-dir ./cogvlm-chat-hf --local-dir-use-symlinks False
```

### 离线推理 pipeline

以下是使用pipeline进行离线推理的示例，更多用法参考[VLM离线推理 pipeline](./vl_pipeline.md)

```python
from lmdeploy import pipeline
from lmdeploy.vl import load_image


if __name__ == "__main__":
    pipe = pipeline('cogvlm-chat-hf')

    image = load_image('https://raw.githubusercontent.com/open-mmlab/mmdeploy/main/tests/data/tiger.jpeg')
    response = pipe(('describe this image', image))
    print(response)
```
