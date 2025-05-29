# PyTorchEngine 性能分析

我们提供了数种分析 PytorchEngine 性能的方式

## PyTorch Profiler

我们集成了 PyTorch Profiler，可以在启动 pipeline 或 api server 时添加环境变量：

```bash
# enable profile cpu
export LMDEPLOY_PROFILE_CPU=1
# enable profile cuda
export LMDEPLOY_PROFILE_CUDA=1
# profile would start after 3 seconds
export LMDEPLOY_PROFILE_DELAY=3
# profile 10 seconds
export LMDEPLOY_PROFILE_DURATION=10
# prefix path to save profile files
export LMDEPLOY_PROFILE_OUT_PREFIX="/path/to/save/profile_"
```

这样在退出程序后，统计信息会被存储在 `LMDEPLOY_PROFILE_OUT_PREFIX` 指定的地址，方便进行性能分析。

## Nsight System

我们也支持使用 Nsight System 分析 nVidia 设备的性能。

### 单卡

单卡情况下比较简单，可以直接使用 `nsys profile`：

```bash
nsys profile python your_script.py
```

### 多卡

当启用了 DP/TP/EP 等多卡方案时，可以设置环境变量

```bash
# enable nsight system
export LMDEPLOY_RAY_NSYS_ENABLE=1
# prefix path to save profile files
export LMDEPLOY_RAY_NSYS_OUT_PREFIX="/path/to/save/profile_"
```

然后正常启动脚本或 api server 即可（注意**不要**添加 `nsys profile`）

这样 profile 的结果就会被保存在 `LMDEPLOY_RAY_NSYS_OUT_PREFIX` 下，如果没有配置 `LMDEPLOY_RAY_NSYS_OUT_PREFIX`，可以在 `/tmp/ray/session_xxx/nsight` 目录下找到。

## Ray timeline

我们使用 ray 实现多卡支持，如果希望查看 ray timeline，可以配置如下环境变量：

```bash
export LMDEPLOY_RAY_TIMELINE_ENABLE=1
export LMDEPLOY_RAY_TIMELINE_OUT_PATH="/path/to/save/timeline.json"
```
