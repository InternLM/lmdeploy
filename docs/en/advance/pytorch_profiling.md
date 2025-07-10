# PyTorchEngine Profiling

We provide multiple profiler to analysis the performance of PyTorchEngine.

## PyTorch Profiler

We have integrated the PyTorch Profiler. You can enable it by setting environment variables when launching the pipeline or API server:

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

After the program exits, the profiling data will be saved to the path specified by `LMDEPLOY_PROFILE_OUT_PREFIX` for performance analysis.

## Nsight System

We also support using Nsight System to profile NVIDIA devices.

### Single GPU

For single-GPU scenarios, simply use `nsys profile`:

```bash
nsys profile python your_script.py
```

### Multi-GPU

When using multi-GPU solutions like DP/TP/EP, set the following environment variables:

```bash
# enable nsight system
export LMDEPLOY_RAY_NSYS_ENABLE=1
# prefix path to save profile files
export LMDEPLOY_RAY_NSYS_OUT_PREFIX="/path/to/save/profile_"
```

Then launch the script or API server as usual (Do **NOT** use nsys profile here).

The profiling results will be saved under `LMDEPLOY_RAY_NSYS_OUT_PREFIX`. If `LMDEPLOY_RAY_NSYS_OUT_PREFIX` is not configured, you can find the results in `/tmp/ray/session_xxx/nsight`.

## Ray timeline

We use `ray` to support multi-device deployment. You can get the ray timeline with the environments below.

```bash
export LMDEPLOY_RAY_TIMELINE_ENABLE=1
export LMDEPLOY_RAY_TIMELINE_OUT_PATH="/path/to/save/timeline.json"
```
