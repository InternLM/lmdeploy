# 常见问题

## ModuleNotFoundError

### No module named 'mmengine.config.lazy'

可能是因为已经有旧版本的mmengine缓存在了本机。更新到最新班应该可以解决这个问题。

```shell
pip install --upgrade mmengine
```

### No module named '\_turbomind'

可能是因为：

1. 您没有安装 lmdeploy 的预编译包。`_turbomind`是 turbomind c++ 的 pybind部分，涉及到编译。推荐您直接安装预编译包。

```
pip install lmdeploy[all]
```

2. 如果已经安装了，还是出现这个问题，请检查下执行目录。不要在 lmdeploy 的源码根目录下执行 python -m lmdeploy.turbomind.\*下的package，换到其他目录下执行。

## Libs

### libnccl.so.2 not found

确保通过 `pip install lmdeploy[all]` 安装了 lmdeploy (>=v0.0.5)。

如果安装之后，问题还存在，那么就把`libnccl.so.2`的路径加入到环境变量 LD_LIBRARY_PATH 中。

```shell
# 获取nvidia-nccl-cu11 package的安装目录
pip show nvidia-nccl-cu11|grep Location
# 把"libnccl.so.2"的路径加入到 LD_LIBRARY_PATH
export LD_LIBRARY_PATH={Location}/nvidia/nccl/lib:$LD_LIBRARY_PATH
```

### symbol cudaFreeAsync version libcudart.so.11.0 not defined in file libcudart.so.11.0 with link time reference

很可能是机器上的 cuda 版本太低导致的。LMDeploy运行时要求 cuda 不低于 11.2

## 推理

### RuntimeError: \[TM\]\[ERROR\] CUDA runtime error: out of memory /workspace/lmdeploy/src/turbomind/utils/allocator.h

通常这是因为 k/v cache内存比例过大导致的。比例的控制参数是 `TurbomindEngineConfig.cache_max_entry_count`。该参数在不同版本的 lmdeploy中，含义略有不同。具体请参考代码中的[演进说明](https://github.com/InternLM/lmdeploy/blob/52419bd5b6fb419a5e3aaf3c3b4dea874b17e094/lmdeploy/messages.py#L107)

如果在使用 pipeline 接口遇到该问题，请调低比例，比如

```python
from lmdeploy import pipeline, TurbomindEngineConfig

backend_config = TurbomindEngineConfig(cache_max_entry_count=0.2)

pipe = pipeline('internlm/internlm2-chat-7b',
                backend_config=backend_config)
response = pipe(['Hi, pls intro yourself', 'Shanghai is'])
print(response)
```

如果在使用 CLI 工具时遇到此问题，请传入参数`--cache-max-entry-count`，调低 k/v cache缓存使用比例。比如，

```shell
# chat 命令
lmdeploy chat turbomind internlm/internlm2-chat-7b --cache-max-entry-count 0.2

# server 命令
lmdeploy serve api_server internlm/internlm2-chat-7b --cache-max-entry-count 0.2
```

## 服务

## 量化

### RuntimeError: \[enforce fail at inline_container.cc:337\] . unexpected pos 4566829760 vs 4566829656

请检查你的硬盘空间。

这个错误是因为保存权重时硬盘空间不足导致的，在量化 70B 模型时可能会遇到

### ModuleNotFoundError: No module named 'flash_attn'

量化 `qwen` 模型需要安装 `flash-attn`。但是，根据社区用户的反馈，`flash-attn` 比较难安装。所以，lmdeploy 从依赖列表中移除 `flash-attn`，用户在用到的时候，可以进行手动安装。
