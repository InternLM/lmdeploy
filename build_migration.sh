# build slime
mkdir -p build/_build_slime; cd build/_build_slime;
cmake -DSLIME_PYTHON_EXECUTABLE=`which python` -DCMAKE_BUILD_TYPE=Release -DUSE_CUDA=ON ../../migration/Slime/
make -j`nproc`;
cp _slime_C.*.so ../../lmdeploy/pytorch/
cd -;

# build mooncake
mkdir -p build/_build_mooncake; cd build/_build_mooncake;
cmake -DCMAKE_BUILD_TYPE=Release -DUSE_CUDA=ON ../../migration/Mooncake/
make -j `nproc`;
cp mooncake-integration/mooncake_lmdeploy_adaptor.*.so ../../lmdeploy/pytorch/
cd -
