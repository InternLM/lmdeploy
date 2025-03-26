# Python Wrapper
find_package(pybind11 REQUIRED)

pybind11_add_module(
    _migration_c
    bind.cpp
)

target_link_libraries(_migration_c PRIVATE _migration)
