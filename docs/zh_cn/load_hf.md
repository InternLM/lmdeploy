# 直接读取 huggingface 模型

在 V0.0.14 版本之前，若想使用 LMDeploy 进行推理或者部署，需要先使用命令 `lmdeploy convert` 将模型离线转换为 TurboMind 推理引擎支持的格式，转换后的模型可以更快地进行加载，但对用户使用来说并不友好，因此，LDMdeploy 决定增加在线转换的功能，支持直接读取 Huggingface 的模型。

## 支持的类型

目前，TurboMind 支持加载三种类型的模型：

1. 在 huggingface.co 上面通过 lmdeploy 量化的模型，如 [llama2-70b-4bit](https://huggingface.co/lmdeploy/llama2-chat-70b-4bit), [internlm-chat-20b-4bit](https://huggingface.co/internlm/internlm-chat-20b-4bit)
2. huggingface.co 上面其他 LM 模型，如Qwen/Qwen-7B-Chat
3. 通过 `lmdeploy convert` 命令转换好的模型，兼容旧格式

## 使用方式

### 1) lmdeploy / internlm 所管理的量化模型

lmdeploy / internlm 所管理的模型，config.json 中已经有在线转换需要的参数，所以使用时只需要传入 repo_id 或者本地路径即可。

> 如果 config.json 还未及时更新，还需要传入`--model-name` 参数，可参考 2)

```
repo_id=lmdeploy/qwen-chat-7b-4bit
# or
# repo_id=/path/to/managed_model

# Inference by TurboMind
lmdeploy chat turbomind $repo_id

# Serving with gradio
lmdeploy serve gradio $repo_id

# Serving with Restful API
lmdeploy serve api_server $repo_id --instance_num 32 --tp 1
```

### 2) 其他的 LM 模型

其他的比较热门的模型比如 Qwen/Qwen-7B-Chat, baichuan-inc/Baichuan2-7B-Chat，需要传入模型的名字。LMDeploy 模型支持情况可通过 `lmdeploy list` 查看。

```
repo_id=Qwen/Qwen-7B-Chat
model_name=qwen-7b
# or
# repo_id=/path/to/Qwen-7B-Chat/local_path

# Inference by TurboMind
lmdeploy chat turbomind $repo_id --model-name $model_name

# Serving with gradio
lmdeploy serve gradio $repo_id --model-name $model_name

# Serving with Restful API
lmdeploy serve api_server $repo_id --model-name $model_name --instance_num 32 --tp 1
```

### 3) 通过 `lmdeploy convert` 命令转换好的模型

使用方式与之前相同

```
# Convert a model
lmdeploy convert /path/to/model ./workspace --model-name MODEL_NAME

# Inference by TurboMind
lmdeploy chat turbomind ./workspace

# Serving with gradio
lmdeploy serve gradio ./workspace

# Serving with Restful API
lmdeploy serve api_server ./workspace --instance_num 32 --tp 1
```
