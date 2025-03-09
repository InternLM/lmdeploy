#include <cstdlib>
#include <ops.h>
#include <pybind11/pybind11.h>
#include <iostream>
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>
#include <iostream>
#include <cstring>

#include "cuda_common.h"
#include "migration_manager.cuh"

#include <torch/extension.h>

std::tuple<std::vector<char>, int64_t> get_ipc_handle(torch::Tensor& kv_tensor) {
    cudaIpcMemHandle_t handle;
    int64_t d_offset = kv_tensor.storage_offset();
    auto d_data = kv_tensor.data_ptr();
    // 创建IPC句柄
    CUDA_CHECK(cudaIpcGetMemHandle(&handle, d_data));

    return std::tuple(serialize_cuda_ipc_handle(handle), d_offset);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  // vLLM custom ops
  pybind11::module ops = m.def_submodule("ops", "migration operators");

  ops.def("init_migration_manager", &init_migration_manager,
          "init migration manager");
  ops.def("migrate", &migrate, "perform migrate from p instance to d instance");
  ops.def("get_ipc_handle", &get_ipc_handle, "get_ipc_handle");

  py::class_<KVCacheHandlerConfig>(m, "KVCacheHandlerConfig")
      .def(py::init<int, int, int, int,
                    std::vector<std::vector<std::vector<std::vector<char>>>> &,
                    std::vector<std::vector<std::vector<int64_t>>> &>())
      .def(py::init<int, int, int, int>())
      .def(py::init<>());
}
