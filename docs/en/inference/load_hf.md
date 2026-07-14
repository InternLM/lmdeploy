# Load huggingface model directly

Starting from v0.1.0, Turbomind adds the ability to pre-process the model parameters on-the-fly while loading them from huggingface style models.

## Supported model type

Currently, Turbomind support loading three types of model:

1. A lmdeploy-quantized model hosted on huggingface.co, such as [llama2-70b-4bit](https://huggingface.co/lmdeploy/llama2-chat-70b-4bit), etc.
2. Other LM models on huggingface.co like Qwen/Qwen2.5-7B-Instruct

## Usage

### 1) A lmdeploy-quantized model

For models quantized by `lmdeploy.lite` such as [llama2-70b-4bit](https://huggingface.co/lmdeploy/llama2-chat-70b-4bit), etc.

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

### 2) Other LM models

For other LM models such as Qwen/Qwen2.5-7B-Instruct or internlm/internlm2-chat-7b. LMDeploy supported models can be viewed through `lmdeploy list`.

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
