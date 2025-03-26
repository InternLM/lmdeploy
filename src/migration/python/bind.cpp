#include "engine/config.h"

#include "engine/rdma_transport.h"
#include "ops/ops.h"
#include "utils/json.hpp"

#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

using json = nlohmann::json;

namespace py = pybind11;

PYBIND11_MODULE(_migration_c, m)
{
    py::class_<migration::RDMAContext>(m, "rdma_context")
        .def(py::init<>())
        .def("init_rdma_context", &migration::RDMAContext::init_rdma_context)
        .def("register_memory", &migration::RDMAContext::register_memory)
        .def("exchange_info", [](migration::RDMAContext& self) { return self.exchange_info().dump(); })
        .def("connect",
             [](migration::RDMAContext& self, std::string exchange_info) {
                 json                json_info       = json::parse(exchange_info);
                 migration::RDMAInfo local_rdma_info = migration::RDMAInfo(json_info["rdma_info"]);
                 self.modify_qp_to_rtsr(local_rdma_info);
                 for (auto& mr_info : json_info["mr_info"].items())
                     self.register_remote_memory(mr_info.key(), mr_info.value());
             })
        .def("cq_poll_handle", &migration::RDMAContext::cq_poll_handle)
        .def("launch_cq_future", &migration::RDMAContext::launch_cq_future)
        .def("stop_cq_future", &migration::RDMAContext::stop_cq_future)
        .def("r_rdma_async",
             &migration::RDMAContext::r_rdma_async,
             py::call_guard<py::gil_scoped_release>(),
             "Read remote memory asynchronously")
        .def("batch_r_rdma_async",
             &migration::RDMAContext::batch_r_rdma_async,
             py::call_guard<py::gil_scoped_release>(),
             "Read remote memory asynchronously");

    m.def("gather", &migration::gather);
    m.def("scatter", &migration::scatter);
}