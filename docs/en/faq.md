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

## Turbomind Inference

## Pytorch Inference

## Serve

## Quantization
