# FAQ

## ModuleNotFoundError

### No module named 'mmengine.config.lazy'

There is probably a cached mmengine in your local host. Try to install its latest version.

```shell
pip install --upgrade mmengine
```

### No module named '\_turbomind'

It may have been caused by the following reasons.

1. You haven't installed lmdeploy's precompiled package. `_turbomind` is the pybind package of c++ turbomind, which involves compilation. It is recommended that you install the precompiled one.

```shell
pip install lmdeploy[all]
```

2. If you have installed it and still encounter this issue, it is probably because you are executing turbomind-related command in the root directory of lmdeploy source code. Switching to another directory will fix it

## Libs

### libnccl.so.2 not found

Make sure you have install lmdeploy (>=v0.0.5) through `pip install lmdeploy[all]`.

If the issue still exists after lmdeploy installation, add the path of `libnccl.so.2` to environment variable LD_LIBRARY_PATH.

```shell
# Get the location of nvidia-nccl-cu11 package
pip show nvidia-nccl-cu11|grep Location
# insert the path of "libnccl.so.2" to LD_LIBRARY_PATH
export LD_LIBRARY_PATH={Location}/nvidia/nccl/lib:$LD_LIBRARY_PATH
```

### symbol cudaFreeAsync version libcudart.so.11.0 not defined in file libcudart.so.11.0 with link time reference

It's probably due to a low-version cuda toolkit. LMDeploy runtime requires a minimum CUDA version of 11.2

## Inference

### RuntimeError: \[TM\]\[ERROR\] CUDA runtime error: out of memory /workspace/lmdeploy/src/turbomind/utils/allocator.h

This is usually due to a disproportionately large memory ratio for the k/v cache, which is dictated by `TurbomindEngineConfig.cache_max_entry_count`.
The implications of this parameter have slight variations in different versions of lmdeploy. For specifics, please refer to the source code for the \[detailed notes\] (https://github.com/InternLM/lmdeploy/blob/52419bd5b6fb419a5e3aaf3c3b4dea874b17e094/lmdeploy/messages.py#L107)

If you encounter this issue while using the pipeline interface, please reduce the `cache_max_entry_count` in `TurbomindEngineConfig` like following:

```python
from lmdeploy import pipeline, TurbomindEngineConfig

backend_config = TurbomindEngineConfig(cache_max_entry_count=0.2)

pipe = pipeline('internlm/internlm2-chat-7b',
                backend_config=backend_config)
response = pipe(['Hi, pls intro yourself', 'Shanghai is'])
print(response)
```

If OOM occurs when you run CLI tools, please pass `--cache-max-entry-count` to decrease k/v cache memory ratio. For example:

```shell
# chat command
lmdeploy chat internlm/internlm2-chat-7b --cache-max-entry-count 0.2

# server command
lmdeploy serve api_server internlm/internlm2-chat-7b --cache-max-entry-count 0.2
```

## Serve

## Quantization

### RuntimeError: \[enforce fail at inline_container.cc:337\] . unexpected pos 4566829760 vs 4566829656

Please check your disk space. This error is due to insufficient disk space when saving weights, which might be encountered when quantizing the 70B model

### ModuleNotFoundError: No module named 'flash_attn'

Quantizing `qwen` requires the installation of `flash-attn`. But based on feedback from community users, `flash-attn` can be challenging to install. Therefore, we have removed it from lmdeploy dependencies and now recommend that users install it it manually as needed.
