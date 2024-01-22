# 从 LMDeploy 创建一个 huggingface 的在线 demo

## 创建 space

首先，注册一个 huggingface 的账号，注册成功后，可以点击右上角头像，选择 New Space 创建。
根据 huggingface 的引导选择需要的配置，完成后即可得到一个空白的 demo。

## 使用 LMDeploy 的 demo

以 `internlm/internlm2-chat-7b` 模型为例，将 space 空间中的`app.py`内容填写为：

```python
from lmdeploy.serve.gradio.turbomind_coupled import run_local
from lmdeploy.messages import TurbomindEngineConfig

backend_config = TurbomindEngineConfig(max_batch_size=1, cache_max_entry_count=0.05)
model_path = 'internlm/internlm2-chat-7b'
run_local(model_path, backend_config=backend_config, huggingface_demo=True)
```

创建`requirements.txt`文本文件，填写如下安装包：

```
lmdeploy
```
