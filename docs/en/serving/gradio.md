# Steps to create a huggingface online demo

## create space

First, register for a Hugging Face account. After successful registration, click on your profile picture in the upper right corner and select “New Space” to create one. Follow the Hugging Face guide to choose the necessary configurations, and you will have a blank demo space ready.

## A demo for LMDeploy

Replace the content of `app.py` in your space with the following code:

```python
from lmdeploy.serve.gradio.turbomind_coupled import run_local
from lmdeploy.messages import TurbomindEngineConfig

backend_config = TurbomindEngineConfig(max_batch_size=1, cache_max_entry_count=0.05)
model_path = 'internlm/internlm2-chat-7b'
run_local(model_path, backend_config=backend_config, server_name="huggingface-space")
```

Create a `requirements.txt` file with the following content:

```
lmdeploy
```

## FAQs

- ZeroGPU compatibility issue. ZeroGPU is more suitable for inference methods similar to PyTorch, rather than Turbomind. You can switch to the PyTorch backend or enable standard GPUs.
- Gradio version issue, versions above 4.0.0 are currently not supported. You can modify this in `app.py`, for example:
  ```python
  import os
  os.system("pip uninstall -y gradio")
  os.system("pip install gradio==3.43.0")
  ```
