mkdir -p _build_migration; cd _build_migration;
cmake -DSLIME_PYTHON_EXECUTABLE=`which python` -DCMAKE_BUILD_TYPE=Release -DUSE_CUDA=ON ../migration/
make -j`nproc`;
cp _C.*.so ../lmdeploy/pytorch
cp Mooncake/mooncake-integration/mooncake_vllm_adaptor.*.so ../lmdeploy/pytorch
cd -;
