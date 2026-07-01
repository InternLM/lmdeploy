# 直接读取 huggingface 模型

从 v0.1.0 开始，Turbomid 添加了直接读取 Huggingface 格式权重的能力。

## 支持的类型

目前，TurboMind 支持加载三种类型的模型：

1. 在 huggingface.co 上面通过 lmdeploy 量化的模型，如 [llama2-70b-4bit](https://huggingface.co/lmdeploy/llama2-chat-70b-4bit)
2. huggingface.co 上面其他 LM 模型，如 Qwen/Qwen2.5-7B-Instruct

## 使用方式

### 1) 通过 lmdeploy 量化的模型

对于通过 `lmdeploy.lite` 量化的模型，TurboMind 可以直接加载，比如 [llama2-70b-4bit](https://huggingface.co/lmdeploy/llama2-chat-70b-4bit).

```
repo_id=lmdeploy/llama2-chat-70b-4bit
model_name=llama2-chat-70b

# or
# repo_id=/path/to/downloaded_model

# Inference by TurboMind
lmdeploy chat $repo_id --model-name $model_name


# Serving with Restful API
lmdeploy serve api_server $repo_id --model-name $model_name --tp 1
```

### 2) 其他的 LM 模型

其他 LM 模型比如 Qwen/Qwen2.5-7B-Instruct, internlm/internlm2-chat-7b。LMDeploy 模型支持情况可通过 `lmdeploy list` 查看。

```
repo_id=Qwen/Qwen2.5-7B-Instruct
model_name=qwen2.5-7b
# or
# repo_id=/path/to/Qwen2.5-7B-Instruct/local_path

# Inference by TurboMind
lmdeploy chat $repo_id --model-name $model_name

# Serving with Restful API
lmdeploy serve api_server $repo_id --model-name $model_name --tp 1
```
