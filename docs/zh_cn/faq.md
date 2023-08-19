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
pip install lmdeploy
```

2. 如果已经安装了，还是出现这个问题，请检查下执行目录。不要在 lmdeploy 的源码根目录下执行 python -m lmdeploy.turbomind.\*下的package，换到其他目录下执行。

## Libs

### libnccl.so.2 not found

确保通过 `pip install lmdeploy` 安装了 lmdeploy (>=v0.0.5)。

如果安装之后，问题还存在，那么就把`libnccl.so.2`的路径加入到环境变量 LD_LIBRARY_PATH 中。

```shell
# 获取nvidia-nccl-cu11 package的安装目录
pip show nvidia-nccl-cu11|grep Location
# 把"libnccl.so.2"的路径加入到 LD_LIBRARY_PATH
export LD_LIBRARY_PATH={Location}/nvidia/nccl/lib:$LD_LIBRARY_PATH
```

### symbol cudaFreeAsync version libcudart.so.11.0 not defined in file libcudart.so.11.0 with link time reference

很可能是机器上的 cuda 版本太低导致的。LMDeploy运行时要求 cuda 不低于 11.2

## Turbomind 推理

## Pytorch 推理

## 服务

## 量化
