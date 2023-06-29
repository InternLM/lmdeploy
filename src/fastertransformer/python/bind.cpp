#include <memory>
#include <pybind11/pybind11.h>
#include "src/fastertransformer/triton_backend/transformer_triton_backend.hpp"

namespace py = pybind11;
namespace ft = fastertransformer;
using namespace pybind11::literals;

PYBIND11_MODULE(_turboformers, m) {
    py::class_<ft::NcclParam>(m, "NcclParam")
        .def(py::init<int, int>(), "rank"_a=0, "world_size"_a=1)
        .def("__str__", &ft::NcclParam::toString);

    py::class_<ft::AbstractCustomComm, std::shared_ptr<ft::AbstractCustomComm>>(m,
        "AbstractCustomComm");

    py::class_<ft::AbstractInstanceComm>(m,
        "AbstractInstanceComm");

    py::class_<AbstractTransformerModel, std::shared_ptr<AbstractTransformerModel>>(m, "AbstractTransformerModel")
        .def_static("create_llama_model", &AbstractTransformerModel::createLlamaModel, "model_dir"_a)
        .def("create_nccl_params", &AbstractTransformerModel::createNcclParams,
            "node_id"_a, "device_id_start"_a=0, "multi_node"_a=false)
        .def("create_custom_comms", [](std::shared_ptr<AbstractTransformerModel>& model, int world_size){
            std::vector<std::shared_ptr<ft::AbstractCustomComm>> ret;
            model->createCustomComms(&ret, world_size);
            return ret;
        }, "world_size"_a)
        .def("create_instance_comm", &AbstractTransformerModel::createInstanceComm, "size"_a);
}