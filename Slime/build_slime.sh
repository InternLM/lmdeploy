mkdir -p build
cd build
cmake ../
make -j`nproc`
cp csrc/python/_slime_c.*.so ../slime/
cd -
pip install -v -e .


