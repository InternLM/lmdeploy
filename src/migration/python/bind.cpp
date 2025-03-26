#include "engine/config.h"

#include "engine/rdma_transport.h"
#include "ops/ops.h"

#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

PYBIND11_MODULE(_migration_c, m)
{
    py::class_<migration::RDMAContext>(m, "rdma_context")
        .def(py::init<>())
        .def("init_rdma_context", &migration::RDMAContext::init_rdma_context)
        .def("register_memory_region", &migration::RDMAContext::registerMemoryRegion)
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
             "Read remote memory asynchronously")
        .def("modify_qp_to_rtsr", &migration::RDMAContext::modify_qp_to_rtsr)
        .def("get_local_rdma_info", &migration::RDMAContext::get_local_rdma_info)
        .def("get_r_key", &migration::RDMAContext::getRKey);

    py::class_<migration::RDMAInfo>(m, "rdma_info")
        .def(py::init<>())
        .def(py::init<uint32_t, uint64_t, uint64_t, int64_t, uint16_t, uint64_t, uint64_t>())
        .def("get_gid", &migration::RDMAInfo::get_gid)
        .def("set_gid", &migration::RDMAInfo::set_gid)
        .def("log", &migration::RDMAInfo::log)
        .def_readwrite("qpn", &migration::RDMAInfo::qpn)
        .def_readwrite("lid", &migration::RDMAInfo::lid)
        .def_readwrite("psn", &migration::RDMAInfo::psn)
        .def_readwrite("mtu", &migration::RDMAInfo::mtu)
        .def_readwrite("gidx", &migration::RDMAInfo::gidx);

    m.def("gather", &migration::gather);
    m.def("scatter", &migration::scatter);
}