# 直接读取 huggingface 模型

从 v0.1.0 开始，Turbomid 添加了直接读取 Huggingface 格式权重的能力。

## 支持的类型

目前，TurboMind 支持加载三种类型的模型：

1. 在 huggingface.co 上面通过 lmdeploy 量化的模型，如 [llama2-70b-4bit](https://huggingface.co/lmdeploy/llama2-chat-70b-4bit), [internlm-chat-20b-4bit](https://huggingface.co/internlm/internlm-chat-20b-4bit)
2. huggingface.co 上面其他 LM 模型，如Qwen/Qwen-7B-Chat
3. 通过 `lmdeploy convert` 命令转换好的模型，兼容旧格式

## 使用方式

### 1) 通过 lmdeploy 量化的模型

对于通过 `lmdeploy.lite` 量化的模型，TurboMind 可以直接加载，比如 [llama2-70b-4bit](https://huggingface.co/lmdeploy/llama2-chat-70b-4bit), [internlm-chat-20b-4bit](https://huggingface.co/internlm/internlm-chat-20b-4bit).

```
repo_id=internlm/internlm-chat-20b-4bit
model_name=internlm-chat-20b

# or
# repo_id=/path/to/downloaded_model

# Inference by TurboMind
lmdeploy chat $repo_id --model-name $model_name

# Serving with gradio
lmdeploy serve gradio $repo_id --model-name $model_name

# Serving with Restful API
lmdeploy serve api_server $repo_id --model-name $model_name --tp 1
```

### 2) 其他的 LM 模型

其他 LM 模型比如 Qwen/Qwen-7B-Chat, baichuan-inc/Baichuan2-7B-Chat。LMDeploy 模型支持情况可通过 `lmdeploy list` 查看。

```
repo_id=Qwen/Qwen-7B-Chat
model_name=qwen-7b
# or
# repo_id=/path/to/Qwen-7B-Chat/local_path

# Inference by TurboMind
lmdeploy chat $repo_id --model-name $model_name

# Serving with gradio
lmdeploy serve gradio $repo_id --model-name $model_name

# Serving with Restful API
lmdeploy serve api_server $repo_id --model-name $model_name --tp 1
```

### 3) 通过 `lmdeploy convert` 命令转换好的模型

使用方式与之前相同

```
# Convert a model
lmdeploy convert $MODEL_NAME /path/to/model --dst-path ./workspace

# Inference by TurboMind
lmdeploy chat ./workspace

# Serving with gradio
lmdeploy serve gradio ./workspace

# Serving with Restful API
lmdeploy serve api_server ./workspace --tp 1
```
