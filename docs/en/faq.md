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
pip install lmdeploy
```

2. If you have installed it and still encounter this issue, it is probably because you are executing turbomind-related command in the root directory of lmdeploy source code. Switching to another directory will fix it

### libnccl.so.2 not found

Make sure you have install lmdeploy (>=v0.0.5) through `pip install lmdeploy`.

If the issue still exists after lmdeploy installation, add the path of `libnccl.so.2` to environment variable LD_LIBRARY_PATH.

```shell
# Get the location of nvidia-nccl-cu11 package
pip show nvidia-nccl-cu11|grep Location
# insert the path of "libnccl.so.2" to LD_LIBRARY_PATH
export LD_LIBRARY_PATH={Location}/nvidia/nccl/lib:$LD_LIBRARY_PATH
```

## Turbomind Inference

## Pytorch Inference

## Serve

## Quantization
