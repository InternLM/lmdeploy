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

```shell
pip install lmdeploy[all]
```

如果您想安装 LMDeploy 预编译包的 nightly 版本，可以根据您的 CUDA 和 Python 版本从 https://github.com/zhyncs/lmdeploy-build 下载并安装最新发布的包。目前更新频率是每天一次。

2. 如果已经安装了，还是出现这个问题，请检查下执行目录。不要在 lmdeploy 的源码根目录下执行 python -m lmdeploy.turbomind.\*下的package，换到其他目录下执行。

但是如果您是开发人员，通常需要在本地进行开发和编译。每次安装 whl 的效率太低了。您可以通过符号链接在编译后指定 lib 的路径。

```shell
# 创建 bld 和进行本地编译
mkdir bld && cd bld && bash ../generate.sh && ninja -j$(nproc)

# 从 bld 中切到 lmdeploy 子目录并设置软链接
cd ../lmdeploy && ln -s ../bld/lib .

# 切换到 lmdeploy 根目录
cd ..

# 使用 python command 比如 check_env
python3 -m lmdeploy check_env
```

如果您仍然遇到在本地机器上找不到 turbomind so 的问题，这意味着您的本地机器上可能存在多个 Python 环境，并且在编译和执行过程中 Python 的版本不匹配。在这种情况下，您需要根据实际情况设置 `lmdeploy/generate.sh` 中的 `PYTHON_EXECUTABLE`，例如 `-DPYTHON_EXECUTABLE=/usr/local/bin/python3`，并且需要重新编译。

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

pipe = pipeline('internlm/internlm2_5-7b-chat',
                backend_config=backend_config)
response = pipe(['Hi, pls intro yourself', 'Shanghai is'])
print(response)
```

如果在使用 CLI 工具时遇到此问题，请传入参数`--cache-max-entry-count`，调低 k/v cache缓存使用比例。比如，

```shell
# chat 命令
lmdeploy chat internlm/internlm2_5-7b-chat --cache-max-entry-count 0.2

# server 命令
lmdeploy serve api_server internlm/internlm2_5-7b-chat --cache-max-entry-count 0.2
```

## 服务

### Api 服务器获取超时

API 服务器的图像 URL 获取超时可通过环境变量 `LMDEPLOY_FETCH_TIMEOUT` 进行配置。默认情况下，请求可能需要长达 10 秒才会超时。

请参阅 [lmdeploy/vl/utils.py](https://github.com/InternLM/lmdeploy/blob/7b6876eafcb842633e0efe8baabe5906d7beeeea/lmdeploy/vl/utils.py#L31) 了解用法。

## 量化

### RuntimeError: \[enforce fail at inline_container.cc:337\] . unexpected pos 4566829760 vs 4566829656

请检查你的硬盘空间。

这个错误是因为保存权重时硬盘空间不足导致的，在量化 70B 模型时可能会遇到

### ModuleNotFoundError: No module named 'flash_attn'

量化 `qwen` 模型需要安装 `flash-attn`。但是，根据社区用户的反馈，`flash-attn` 比较难安装。所以，lmdeploy 从依赖列表中移除 `flash-attn`，用户在用到的时候，可以进行手动安装。
