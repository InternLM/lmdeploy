# Serving with Gradio

Starting an LLM model's gradio service with LMDeploy and interacting with the model on the WebUI is incredibly simple.

```shell
pip install lmdeploy[serve]
lmdeploy serve gradio {model_path}
```

All it takes is one-line command, with the `{model_path}` replaced by the model ID from huggingface hub, such as `internlm/internlm2-chat-7b`, or the local path to the model.

For detailed parameters of the command, please turn to `lmdeploy serve gradio -h` for help.

## Create a huggingface demo

If you want to create an online demo project for your model on huggingface, please follow the steps below.

## Step 1: Create space

First, register for a Hugging Face account. After successful registration, click on your profile picture in the upper right corner and select “New Space” to create one. Follow the Hugging Face guide to choose the necessary configurations, and you will have a blank demo space ready.

## Step 2: Develop demo's entrypoint `app.py`

Replace the content of `app.py` in your space with the following code:

```python
from lmdeploy.serve.gradio.turbomind_coupled import run_local
from lmdeploy.messages import TurbomindEngineConfig

backend_config = TurbomindEngineConfig(max_batch_size=8)
model_path = 'internlm/internlm2-chat-7b'
run_local(model_path, backend_config=backend_config, server_name="huggingface-space")
```

Create a `requirements.txt` file with the following content:

```
lmdeploy
```

## FAQs

- ZeroGPU compatibility issue. ZeroGPU is not suitable for LMDeploy turbomind engine. Please use the standard GPUs. Or, you can change the backend config in the above code to `PyTorchEngineConfig` to use the ZeroGPU.
- Gradio version issue, versions above 4.0.0 are currently not supported. You can modify this in `app.py`, for example:
  ```python
  import os
  os.system("pip uninstall -y gradio")
  os.system("pip install gradio==3.43.0")
  ```
