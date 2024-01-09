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

## Turbomind Inference

## Pytorch Inference

## Serve

## Quantization

### RuntimeError: \[enforce fail at inline_container.cc:337\] . unexpected pos 4566829760 vs 4566829656

Please check your disk space. This error is due to insufficient disk space when saving weights, which might be encountered when quantizing the 70B model

### ModuleNotFoundError: No module named 'flash_attn'

Quantizing `qwen` requires the installation of `flash-attn`. But based on feedback from community users, `flash-attn` can be challenging to install. Therefore, we have removed it from lmdeploy dependencies and now recommend that users install it it manually as needed.
