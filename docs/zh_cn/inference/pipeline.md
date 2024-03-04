# LLM 离线推理 pipeline

本文通过一些例子展示 pipeline 的基本用法。

pipeline API 详细的接口说明，请阅读[此处](https://lmdeploy.readthedocs.io/zh-cn/latest/api/pipeline.html)

## 使用方法

- **使用默认参数的例子:**

```python
from lmdeploy import pipeline

pipe = pipeline('internlm/internlm2-chat-7b')
response = pipe(['Hi, pls intro yourself', 'Shanghai is'])
print(response)
```

在这个例子中，pipeline 默认申请一定比例显存，用来存储推理过程中产生的 k/v。比例由参数 `TurbomindEngineConfig.cache_max_entry_count` 控制。

LMDeploy 在研发过程中，k/v cache 比例的设定策略有变更，以下为变更记录：

1. `v0.2.0 <= lmdeploy <= v0.2.1`

   默认比例为 0.5，表示 **GPU总显存**的 50% 被分配给 k/v cache。 对于 7B 模型来说，如果显存小于 40G，会出现 OOM。当遇到 OOM 时，请按照下面的方法，酌情降低 k/v cache 占比：

   ```python
   from lmdeploy import pipeline, TurbomindEngineConfig

   # 调低 k/v cache内存占比调整为总显存的 20%
   backend_config = TurbomindEngineConfig(cache_max_entry_count=0.2)

   pipe = pipeline('internlm/internlm2-chat-7b',
                   backend_config=backend_config)
   response = pipe(['Hi, pls intro yourself', 'Shanghai is'])
   print(response)
   ```

2. `lmdeploy > v0.2.1`

   分配策略改为从**空闲显存**中按比例为 k/v cache 开辟空间。默认比例值调整为 0.8。如果遇到 OOM，类似上面的方法，请酌情减少比例值，降低 k/v cache 的内存占用量

- **如何设置 tp:**

```python
from lmdeploy import pipeline, TurbomindEngineConfig

backend_config = TurbomindEngineConfig(tp=2)
pipe = pipeline('internlm/internlm2-chat-7b',
                backend_config=backend_config)
response = pipe(['Hi, pls intro yourself', 'Shanghai is'])
print(response)
```

- **如何设置 sampling 参数:**

```python
from lmdeploy import pipeline, GenerationConfig, TurbomindEngineConfig

backend_config = TurbomindEngineConfig(tp=2)
gen_config = GenerationConfig(top_p=0.8,
                              top_k=40,
                              temperature=0.8,
                              max_new_tokens=1024)
pipe = pipeline('internlm/internlm2-chat-7b',
                backend_config=backend_config)
response = pipe(['Hi, pls intro yourself', 'Shanghai is'],
                gen_config=gen_config)
print(response)
```

- **如何设置 OpenAI 格式输入:**

```python
from lmdeploy import pipeline, GenerationConfig, TurbomindEngineConfig

backend_config = TurbomindEngineConfig(tp=2)
gen_config = GenerationConfig(top_p=0.8,
                              top_k=40,
                              temperature=0.8,
                              max_new_tokens=1024)
pipe = pipeline('internlm/internlm2-chat-7b',
                backend_config=backend_config)
prompts = [[{
    'role': 'user',
    'content': 'Hi, pls intro yourself'
}], [{
    'role': 'user',
    'content': 'Shanghai is'
}]]
response = pipe(prompts,
                gen_config=gen_config)
print(response)
```

- **流式返回处理结果：**

```python
from lmdeploy import pipeline, GenerationConfig, TurbomindEngineConfig

backend_config = TurbomindEngineConfig(tp=2)
gen_config = GenerationConfig(top_p=0.8,
                              top_k=40,
                              temperature=0.8,
                              max_new_tokens=1024)
pipe = pipeline('internlm/internlm2-chat-7b',
                backend_config=backend_config)
prompts = [[{
    'role': 'user',
    'content': 'Hi, pls intro yourself'
}], [{
    'role': 'user',
    'content': 'Shanghai is'
}]]
for item in pipe.stream_infer(prompts, gen_config=gen_config):
    print(item)
```

- **使用 pytorch 后端**

需要先安装 triton

```shell
pip install triton>=2.1.0
```

```python
from lmdeploy import pipeline, GenerationConfig, PytorchEngineConfig

backend_config = PytorchEngineConfig(session_len=2048)
gen_config = GenerationConfig(top_p=0.8,
                              top_k=40,
                              temperature=0.8,
                              max_new_tokens=1024)
pipe = pipeline('internlm/internlm2-chat-7b',
                backend_config=backend_config)
prompts = [[{
    'role': 'user',
    'content': 'Hi, pls intro yourself'
}], [{
    'role': 'user',
    'content': 'Shanghai is'
}]]
response = pipe(prompts, gen_config=gen_config)
print(response)
```

## FAQs

- **RuntimeError: An attempt has been made to start a new process before the current process has finished its bootstrapping phase**.

  如果你在使用 tp>1 和 pytorch 后端的时候，遇到了这个错误。请确保 python 脚本中有下面内容作为入口

  ```python
  if __name__ == '__main__':
  ```

  一般来说，在多线程或多进程上下文中，可能需要确保初始化代码只执行一次。这时候，`if __name__ == '__main__':` 可以帮助确保这些初始化代码只在主程序执行，而不会在每个新创建的进程或线程中重复执行。
