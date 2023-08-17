# Build lmdeploy on windows

## Requirements

- [CMake 3.17+](https://github.com/Kitware/CMake/releases)
- [Visual Studio 2019+](https://visualstudio.microsoft.com/downloads/)
- [CUDA Toolkit 11.8+](https://developer.nvidia.com/cuda-toolkit-archive)

## Build lmdeploy wheel

```powershell
mkdir build
cd build
..\builder\windows\generate.ps1
cmake --build . --config Release -- /m
cmake --install . --config Release
cd ..
rm build -Force -Recurse
python setup.py bdist_wheel -d build\wheel
```
