# Build lmdeploy on windows

## Requirements

- [CMake 3.17+](https://github.com/Kitware/CMake/releases)
- [Visual Studio 2019+](https://visualstudio.microsoft.com/downloads/)
- [CUDA Toolkit 11.8+](https://developer.nvidia.com/cuda-toolkit-archive)

## Build lmdeploy wheel

```powershell
pip install build
python -m build --wheel
```
