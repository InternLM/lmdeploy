# Build lmdeploy on windows

## Requirements

- [CMake 3.25.2+](https://github.com/Kitware/CMake/releases)
- [Visual Studio 2022](https://visualstudio.microsoft.com/downloads/), with a toolset supported by the selected CUDA Toolkit
- [CUDA Toolkit 12.0+](https://developer.nvidia.com/cuda-toolkit-archive)

TurboMind requires C++20 for both host C++ and CUDA compilation.

CMake applies `/Zc:twoPhase-` to NVCC's host pass to work around CuTe template parsing with MSVC.

## Build lmdeploy wheel

```powershell
pip install build
python -m build --wheel
```
