#pragma once

#include <pybind11/pybind11.h>

namespace turbomind::python {

void BindSpeculativeSampling(pybind11::module_& module);

void BindDraftCarry(pybind11::module_& module);

void BindTargetHiddenProjection(pybind11::module_& module);

void BindSpeculativeSequence(pybind11::module_& module);

}  // namespace turbomind::python
