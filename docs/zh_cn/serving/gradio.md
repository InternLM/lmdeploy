# 部署 gradio 服务

通过 LMDeploy 启动 LLM 模型的 gradio 服务，并在 WebUI 上和模型对话特别简单，一条命令即可。

```shell
pip install lmdeploy[serve]
lmdeploy serve gradio {model_path}
```

把上面命令中的 `{model_path}` 换成 huggingface hub 上的模型 id，比如 internlm/internlm2_5-7b-chat，或者换成模型的本地路径就可以了。

关于命令的详细参数，请使用 `lmdeploy serve gradio --help` 查阅。

## 创建 huggingface demo

如果想要在 huggingface 上创建模型的在线演示项目，请按以下步骤进行。

### 第一步：创建 space

首先，注册一个 huggingface 的账号，注册成功后，可以点击右上角头像，选择 New Space 创建。
根据 huggingface 的引导选择需要的配置，完成后即可得到一个空白的 demo。

### 第二步：编写 demo 入口代码 app.py

以 `internlm/internlm2_5-7b-chat` 模型为例，将 space 空间中的`app.py`内容填写为：

```python
from lmdeploy.serve.gradio.turbomind_coupled import run_local
from lmdeploy.messages import TurbomindEngineConfig

backend_config = TurbomindEngineConfig(max_batch_size=8)
model_path = 'internlm/internlm2_5-7b-chat'
run_local(model_path, backend_config=backend_config, server_name="huggingface-space")
```

创建`requirements.txt`文本文件，填写如下安装包：

```
lmdeploy
```

## FAQs

- ZeroGPU 适配问题。ZeroGPU不适用 LMDeploy Turbomind 引擎，请选择普通 GPU，或者把上述代码中的 backend_config 改成 PyTorchEngineConfig，就可以用 ZeroGPU 了。
- gradio 版本问题，目前不支持 4.0.0 以上版本，可以在 `app.py` 中修改，类似：
  ```python
  import os
  os.system("pip uninstall -y gradio")
  os.system("pip install gradio==3.43.0")
  ```
