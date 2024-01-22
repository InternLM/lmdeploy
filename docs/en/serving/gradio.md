# Steps to create a huggingface online demo

## create space

First, register for a Hugging Face account. After successful registration, click on your profile picture in the upper right corner and select “New Space” to create one. Follow the Hugging Face guide to choose the necessary configurations, and you will have a blank demo space ready.

## 使用 LMDeploy 的 demo

Replace the content of `app.py` in your space with the following code:

```python
from lmdeploy.serve.gradio.turbomind_coupled import run_local
from lmdeploy.messages import TurbomindEngineConfig

backend_config = TurbomindEngineConfig(max_batch_size=1, cache_max_entry_count=0.05)
model_path = 'internlm/internlm2-chat-7b'
run_local(model_path, backend_config=backend_config, huggingface_demo=True)
```

Create a `requirements.txt` file with the following content:

```
lmdeploy
```
