#include "core/graph_handler.h"
#include <pybind11/stl.h>

#ifdef USE_CUDA
#include "cuda/operator_timer.h"
#endif

namespace py = pybind11;

namespace infini {

using namespace py::literals;
using policy = py::return_value_policy;

void register_operator_timer(py::module &m) {
#ifdef USE_CUDA
    using namespace opTimer;
    m.def("getPerfConvCudnn", &getPerfConvCudnn);
    m.def("getPerfConvTransposed2dCudnn", &getPerfConvTransposed2dCudnn);
    m.def("getPerfMatmulCublas", &getPerfMatmulCublas);
#endif
}

void init_graph_builder(py::module &m) {
    m.def("cpu_runtime", &CpuRuntimeObj::getInstance);
    py::class_<RuntimeObj, std::shared_ptr<RuntimeObj>>(m, "RuntimeObj");
    py::class_<CpuRuntimeObj, std::shared_ptr<CpuRuntimeObj>, RuntimeObj>(
        m, "CpuRuntimeObj");
    py::class_<Shape>(m, "Shape");
    py::class_<TensorObj, std::shared_ptr<TensorObj>>(m, "TensorObj");
    py::class_<Tensor>(m, "Tensor");
    py::enum_<ActType>(m, "ActType")
        .value("Linear", ActType::None) // None 是 Python 关键字，不能用
        .value("Relu", ActType::Relu)
        .value("Sigmoid", ActType::Sigmoid)
        .value("Tanh", ActType::Tanh)
        .export_values();
    py::class_<GraphHandler>(m, "GraphHandler");
    py::class_<GraphHandlerObj>(m, "GraphHandlerObj")
        .def(py::init<Runtime>())
        .def("tensor", py::overload_cast<Shape, int>(&GraphHandlerObj::tensor),
             policy::reference_internal)
        .def("matmul",
             py::overload_cast<Tensor, Tensor, Tensor, bool, bool, Tensor,
                               ActType>(&GraphHandlerObj::matmul),
             policy::move)
        .def("add",
             py::overload_cast<Tensor, Tensor, Tensor>(&GraphHandlerObj::add),
             policy::move)
        .def("sub",
             py::overload_cast<Tensor, Tensor, Tensor>(&GraphHandlerObj::sub),
             policy::move)
        .def("mul",
             py::overload_cast<Tensor, Tensor, Tensor>(&GraphHandlerObj::mul),
             policy::move)
        .def("div",
             py::overload_cast<Tensor, Tensor, Tensor>(&GraphHandlerObj::div),
             policy::move)
        .def("pow",
             py::overload_cast<Tensor, Tensor, Tensor>(&GraphHandlerObj::pow),
             policy::move)
        .def("relu", py::overload_cast<Tensor, Tensor>(&GraphHandlerObj::relu),
             policy::move)
        .def("sigmoid",
             py::overload_cast<Tensor, Tensor>(&GraphHandlerObj::sigmoid),
             policy::move)
        .def("tanh", py::overload_cast<Tensor, Tensor>(&GraphHandlerObj::tanh),
             policy::reference_internal)
        .def("softmax",
             py::overload_cast<Tensor, Tensor>(&GraphHandlerObj::softmax),
             policy::move)
        .def("abs", py::overload_cast<Tensor, Tensor>(&GraphHandlerObj::abs),
             policy::move);
}

} // namespace infini

PYBIND11_MODULE(backend, m) {
    infini::register_operator_timer(m);
    infini::init_graph_builder(m);
}
