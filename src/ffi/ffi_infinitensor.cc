#include <pybind11/stl.h>
#ifdef USE_CUDA
#include "cuda/operator_timer.h"
#endif
#include "core/graph_factory.h"
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

void init_graph_factory(py::module &m) {
    py::class_<TensorObj>(m, "TensorObj");
    py::class_<Tensor>(m, "Tensor");
    py::class_<OperatorObj>(m, "OperatorObj");
    py::class_<Operator>(m, "Operator");
    py::class_<GraphObj>(m, "GraphObj");
    py::class_<Graph>(m, "Graph");
    py::class_<TensorVec>(m, "TensorVec");
    py::class_<GraphFactory>(m, "GraphFactory");
    py::class_<GraphFactoryObj>(m, "GraphFactoryObj")
        .def(py::init<Runtime>())
        .def("conv",
             py::overload_cast<Tensor, Tensor, Tensor, int, int, int, int, int,
                               int, Tensor>(&GraphFactoryObj::conv),
             "input"_a, "weight"_a, "output"_a, "ph"_a, "pw"_a, "sh"_a = 1,
             "sw"_a = 1, "dh"_a = 1, "dw"_a = 1, "bias"_a = nullptr,
             policy::reference_internal)
        .def("matmul",
             py::overload_cast<Tensor, Tensor, Tensor, bool, bool, Tensor,
                               ActType>(&GraphFactoryObj::matmul),
             policy::reference_internal)
        .def("convTrans",
             py::overload_cast<Tensor, Tensor, Tensor, int, int, int, int, int,
                               int, int, int, int, Tensor, ActType>(
                 &GraphFactoryObj::convTrans),
             policy::reference_internal)
        .def("g2bmm",
             py::overload_cast<Tensor, Tensor, Tensor, const int, const int,
                               Tensor, ActType>(&GraphFactoryObj::g2bmm),
             policy::reference_internal)
        .def("gbmml",
             py::overload_cast<Tensor, Tensor, Tensor, const int, Tensor,
                               ActType>(&GraphFactoryObj::gbmml),
             policy::reference_internal)

        .def("pad",
             py::overload_cast<Tensor, Tensor, const vector<int> &,
                               const optional<const vector<int>> &>(
                 &GraphFactoryObj::pad),
             policy::reference_internal)
     //    .def("slice",
     //         py::overload_cast<
     //             Tensor, Tensor, const vector<int> &, const vector<int> &,
     //             const optional< const vector<int> > &,
     //             const optional< const vector<int> > &>(
     //                &GraphFactoryObj::slice),
     //         policy::reference_internal)
        .def(
            "concat",
            py::overload_cast<TensorVec, Tensor, int>(&GraphFactoryObj::concat),
            policy::reference_internal)
        .def("split",
             py::overload_cast<Tensor, std::optional<TensorVec>, int, int>(
                 &GraphFactoryObj::split),
             policy::reference_internal)
        .def("extend",
             py::overload_cast<Tensor, Tensor, int, int>(
                 &GraphFactoryObj::extend),
             policy::reference_internal)
        .def("maxpool",
             py::overload_cast<Tensor, Tensor, int, int, int, int, int, int,
                               int, int>(&GraphFactoryObj::maxpool),
             policy::reference_internal)
        .def("avgpool",
             py::overload_cast<Tensor, Tensor, int, int, int, int, int, int,
                               int, int>(&GraphFactoryObj::avgpool),
             policy::reference_internal)
        .def("add",
             py::overload_cast<Tensor, Tensor, Tensor>(&GraphFactoryObj::add),
             policy::reference_internal)
        .def("add",
             py::overload_cast<Tensor, Tensor, Tensor>(&GraphFactoryObj::add),
             policy::reference_internal)
        .def("sub",
             py::overload_cast<Tensor, Tensor, Tensor>(&GraphFactoryObj::sub),
             policy::reference_internal)
        .def("mul",
             py::overload_cast<Tensor, Tensor, Tensor>(&GraphFactoryObj::mul),
             policy::reference_internal)
        .def("div",
             py::overload_cast<Tensor, Tensor, Tensor>(&GraphFactoryObj::div),
             policy::reference_internal)
        .def("pow",
             py::overload_cast<Tensor, Tensor, Tensor>(&GraphFactoryObj::pow),
             policy::reference_internal)
        .def("gather",
             py::overload_cast<Tensor, Tensor, Tensor, int>(
                 &GraphFactoryObj::gather),
             policy::reference_internal)
        .def("reshape",
             py::overload_cast<Tensor, Tensor, const Shape &>(
                 &GraphFactoryObj::reshape),
             policy::reference_internal)
        .def("flatten",
             py::overload_cast<Tensor, Tensor>(&GraphFactoryObj::flatten),
             policy::reference_internal)
        .def("identity",
             py::overload_cast<Tensor, Tensor>(&GraphFactoryObj::identity),
             policy::reference_internal)
        .def("softmax",
             py::overload_cast<Tensor, Tensor>(&GraphFactoryObj::softmax),
             policy::reference_internal)
        .def("relu", py::overload_cast<Tensor, Tensor>(&GraphFactoryObj::relu),
             policy::reference_internal)
        .def("sigmoid",
             py::overload_cast<Tensor, Tensor>(&GraphFactoryObj::sigmoid),
             policy::reference_internal)
        .def("tanh", py::overload_cast<Tensor, Tensor>(&GraphFactoryObj::tanh),
             policy::reference_internal)
        .def("abs", py::overload_cast<Tensor, Tensor>(&GraphFactoryObj::abs),
             policy::reference_internal)
        .def("memBound",
             py::overload_cast<const TensorVec &, const TensorVec &,
                               const std::vector<nnet::Tensor> &, nnet::Expr,
                               double, std::string>(&GraphFactoryObj::memBound),
             policy::reference_internal);
}

} // namespace infini

PYBIND11_MODULE(pyinfinitensor, m) {
    infini::register_operator_timer(m);
    infini::init_graph_factory(m);
}