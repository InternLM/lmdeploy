# cogvlm

## 简介

CogVLM 是一个强大的开源视觉语言模型（VLM）. LMDeploy 已在PyTorch后端支持 CogVLM-17B 模型 [THUDM/cogvlm-chat-hf](https://huggingface.co/THUDM/cogvlm-chat-hf) 和 CogVLM2-19B 模型如[THUDM/cogvlm2-llama3-chat-19B](https://huggingface.co/THUDM/cogvlm2-llama3-chat-19B)

## 快速开始

### 安装 lmdeploy

使用 pip(Python 3.8+)安装LMDeploy，更多安装方式参考 [安装](https://lmdeploy.readthedocs.io/zh-cn/latest/get_started.html#id2)。

```shell
pip install lmdeploy
```

### 准备模型

使用`huggingface-cli`下载 CogVLM 模型

```shell
# 下载模型
huggingface-cli download THUDM/cogvlm-chat-hf --local-dir ./cogvlm-chat-hf --local-dir-use-symlinks False
# cogvlm-chat-hf 使用lmsys/vicuna-7b-v1.5的分词器，需要先将相关文件下载到模型目录中。但对于CogVLM2模型，可跳过此步骤
huggingface-cli download lmsys/vicuna-7b-v1.5 special_tokens_map.json tokenizer.model tokenizer_config.json --local-dir ./cogvlm-chat-hf --local-dir-use-symlinks False
```

### 离线推理 pipeline

以下是使用pipeline进行离线推理的示例，更多用法参考[VLM离线推理 pipeline](https://lmdeploy.readthedocs.io/zh-cn/latest/inference/vl_pipeline.html#vlm-pipeline)

```python
from lmdeploy import pipeline, PytorchEngineConfig
from lmdeploy.vl import load_image

pipe = pipeline('cogvlm-chat-hf', model_name='cogvlm', backend_config=PytorchEngineConfig(tp=1, max_prefill_token_num=4096, cache_max_entry_count=0.8))

image = load_image('https://raw.githubusercontent.com/open-mmlab/mmdeploy/main/tests/data/tiger.jpeg')
response = pipe(('describe this image', image))
print(response)
```
