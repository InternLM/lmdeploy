cmake .. -A x64 -T "v143,cuda=$env:CUDA_PATH" `
    -DCMAKE_BUILD_TYPE=Release `
    -DCMAKE_INSTALL_PREFIX=install `
    -DBUILD_PY_FFI=ON `
    -DBUILD_MULTI_GPU=OFF `
    -DUSE_NVTX=OFF `
    -DBUILD_TEST="$env:BUILD_TEST"
