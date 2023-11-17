# Load huggingface model directly

Before v0.0.14, if you want to serving or inference by TurboMind, you should first convert the model to TurboMind format. Through offline conversion, the model can be loaded faster, but it isn't user-friendly. Therefore, LMDeploy adds the ability of online conversion and support loading huggingface model directly.

## Supported model type

Currently, Turbomind support loading three types of model:

1. A lmdeploy-quantized model hosted on huggingface.co, such as [llama2-70b-4bit](https://huggingface.co/lmdeploy/llama2-chat-70b-4bit), [internlm-chat-20b-4bit](https://huggingface.co/internlm/internlm-chat-20b-4bit), etc.
2. Other hot LM models on huggingface.co like Qwen/Qwen-7B-Chat
3. A model converted by `lmdeploy convert`, old format

## Usage

### 1) A quantized model managed by lmdeploy / internlm

For quantized models managed by lmdeploy or internlm, the parameters required for online conversion are already exist in config.json, so you only need to pass the repo_id or local path when using it.

> If config.json has not been updated in time, you need to pass the `--model-name` parameter, please refer to 2)

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

### 2) Other hot LM models

For other popular models such as Qwen/Qwen-7B-Chat or baichuan-inc/Baichuan2-7B-Chat, the name of the model needs to be passed in. LMDeploy supported models can be viewed through `lmdeploy list`.

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

### 3) A model converted by `lmdeploy convert`

The usage is like previous

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
