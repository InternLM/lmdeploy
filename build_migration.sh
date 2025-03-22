mkdir -p _migration_build
cd _migration_build
cmake ../src/migration
make -j`nproc`
cp _migration_c.*.so ../lmdeploy/migration/
