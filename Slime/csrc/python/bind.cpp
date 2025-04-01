#include "engine/config.h"

#include "engine/rdma_transport.h"
#include "ops/ops.h"
#include "utils/json.hpp"
#include "utils/logging.h"
#include "utils/utils.h"

#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

using json = nlohmann::json;

namespace py = pybind11;

PYBIND11_MODULE(_slime_c, m)
{
    py::class_<slime::RDMAContext>(m, "rdma_context")
        .def(py::init<>())
        .def("init_rdma_context", &slime::RDMAContext::init_rdma_context)
        .def("register_memory_region", &slime::RDMAContext::register_memory_region)
        .def("register_remote_memory_region",
             [](slime::RDMAContext& self, std::string mr_info) {
                 json json_info = json::parse(mr_info);
                 for (auto& item : json_info["mr_info"].items())
                     self.register_remote_memory_region(item.key(), item.value());
             })
        .def("local_info", [](slime::RDMAContext& self) { return self.local_info().dump(); })
        .def("connect",
             [](slime::RDMAContext& self, std::string exchange_info) {
                 json            json_info       = json::parse(exchange_info);
                 slime::RDMAInfo local_rdma_info = slime::RDMAInfo(json_info["rdma_info"]);
                 self.modify_qp_to_rtsr(local_rdma_info);
                 for (auto& mr_info : json_info["mr_info"].items())
                     self.register_remote_memory_region(mr_info.key(), mr_info.value());
             })
        .def("cq_poll_handle", &slime::RDMAContext::cq_poll_handle)
        .def("launch_cq_future", &slime::RDMAContext::launch_cq_future)
        .def("stop_cq_future", &slime::RDMAContext::stop_cq_future)
        .def("r_rdma_async",
             &slime::RDMAContext::r_rdma_async,
             py::call_guard<py::gil_scoped_release>(),
             "Read remote memory asynchronously")
        .def("batch_r_rdma_async",
             &slime::RDMAContext::batch_r_rdma_async,
             py::call_guard<py::gil_scoped_release>(),
             "Read remote memory asynchronously")
        .def("send_async", &slime::RDMAContext::send_async)
        .def("recv_async", &slime::RDMAContext::recv_async);

    m.def("avaliable_nic", &slime::avaliable_nic);

    m.def("gather", &slime::gather);
    m.def("scatter", &slime::scatter);
}
