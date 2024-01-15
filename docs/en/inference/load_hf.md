# Load huggingface model directly

Starting from v0.1.0, Turbomind adds the ability to pre-process the model parameters on-the-fly while loading them from huggingface style models.

## Supported model type

Currently, Turbomind support loading three types of model:

1. A lmdeploy-quantized model hosted on huggingface.co, such as [llama2-70b-4bit](https://huggingface.co/lmdeploy/llama2-chat-70b-4bit), [internlm-chat-20b-4bit](https://huggingface.co/internlm/internlm-chat-20b-4bit), etc.
2. Other LM models on huggingface.co like Qwen/Qwen-7B-Chat
3. A model converted by `lmdeploy convert`, legacy format

## Usage

### 1) A lmdeploy-quantized model

For models quantized by `lmdeploy.lite` such as [llama2-70b-4bit](https://huggingface.co/lmdeploy/llama2-chat-70b-4bit), [internlm-chat-20b-4bit](https://huggingface.co/internlm/internlm-chat-20b-4bit), etc.

```
repo_id=internlm/internlm-chat-20b-4bit
model_name=internlm-chat-20b
# or
# repo_id=/path/to/downloaded_model

# Inference by TurboMind
lmdeploy chat turbomind $repo_id --model-name $model_name

# Serving with gradio
lmdeploy serve gradio $repo_id --model-name $model_name

# Serving with Restful API
lmdeploy serve api_server $repo_id --model-name $model_name --tp 1
```

### 2) Other LM models

For other LM models such as Qwen/Qwen-7B-Chat or baichuan-inc/Baichuan2-7B-Chat. LMDeploy supported models can be viewed through `lmdeploy list`.

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
lmdeploy serve api_server $repo_id --model-name $model_name --tp 1
```

### 3) A model converted by `lmdeploy convert`

The usage is like previous

```
# Convert a model
lmdeploy convert $MODEL_NAME /path/to/model --dst-path ./workspace

# Inference by TurboMind
lmdeploy chat turbomind ./workspace

# Serving with gradio
lmdeploy serve gradio ./workspace

# Serving with Restful API
lmdeploy serve api_server ./workspace --tp 1
```
