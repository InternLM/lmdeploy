# Python Wrapper
set(PYBIND11_FINDPYTHON ON)  # 强制使用 FindPython
find_package(pybind11 REQUIRED)

pybind11_add_module(
    _slime_c
    bind.cpp
)

target_link_libraries(_slime_c PRIVATE _slime)
