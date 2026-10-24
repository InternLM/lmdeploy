#pragma once

#include <pybind11/pybind11.h>

namespace turbomind::python {

void BindVerificationAttention(pybind11::module_& module);

}  // namespace turbomind::python
