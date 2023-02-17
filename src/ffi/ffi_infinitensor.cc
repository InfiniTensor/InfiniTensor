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

void init_values(py::module &m) {
#define VALUE(TYPE, NAME) value(#NAME, TYPE::NAME)

    py::enum_<ActType>(m, "ActType")
        .value("Linear", ActType::None) // None 是 Python 关键字，不能用
        .VALUE(ActType, Relu)
        .VALUE(ActType, Sigmoid)
        .VALUE(ActType, Tanh)
        .export_values();

    py::enum_<OpType>(m, "OpType")
        .VALUE(OpType, Unknown)
        .VALUE(OpType, Conv)
        .VALUE(OpType, Matmul)
        .VALUE(OpType, ConvTrans)
        .VALUE(OpType, G2BMM)
        .VALUE(OpType, GBMM)
        .VALUE(OpType, Pad)
        .VALUE(OpType, Slice)
        .VALUE(OpType, Concat)
        .VALUE(OpType, Split)
        .VALUE(OpType, Transpose)
        .VALUE(OpType, Extend)
        .VALUE(OpType, MaxPool)
        .VALUE(OpType, AvgPool)
        .VALUE(OpType, Add)
        .VALUE(OpType, Sub)
        .VALUE(OpType, Mul)
        .VALUE(OpType, Div)
        .VALUE(OpType, Pow)
        .VALUE(OpType, Gather)
        .VALUE(OpType, ReduceMean)
        .VALUE(OpType, Reshape)
        .VALUE(OpType, Flatten)
        .VALUE(OpType, Identity)
        .VALUE(OpType, BatchNorm)
        .VALUE(OpType, Softmax)
        .VALUE(OpType, Activation)
        .VALUE(OpType, Relu)
        .VALUE(OpType, Sigmoid)
        .VALUE(OpType, Tanh)
        .VALUE(OpType, Abs)
        .VALUE(OpType, Resize)
        .VALUE(OpType, MemBound)
        .export_values();

#undef VALUE
}

void init_graph_builder(py::module &m) {
    using Handler = GraphHandlerObj;

    m.def("cpu_runtime", &CpuRuntimeObj::getInstance);
    py::class_<RuntimeObj, std::shared_ptr<RuntimeObj>>(m, "Runtime");
    py::class_<CpuRuntimeObj, std::shared_ptr<CpuRuntimeObj>, RuntimeObj>(
        m, "CpuRuntime");
    py::class_<TensorObj, std::shared_ptr<TensorObj>>(m, "TensorObj")
        .def("src", &TensorObj::getOutputOf, policy::move);
    py::class_<OperatorObj, std::shared_ptr<OperatorObj>>(m, "Operator")
        .def("op_type", &OperatorObj::getOpType, policy::move);
    py::class_<Handler>(m, "GraphHandler")
        .def(py::init<Runtime>())
        .def("tensor", py::overload_cast<Shape, int>(&Handler::tensor),
             policy::move)
        .def("conv", &Handler::conv, policy::move)
        .def("matmul",
             py::overload_cast<Tensor, Tensor, Tensor, bool, bool, Tensor,
                               ActType>(&Handler::matmul),
             policy::move)
        .def("batchNorm",
             py::overload_cast<Tensor, Tensor, Tensor, Tensor, Tensor, Tensor,
                               float, float, bool>(&Handler::batchNorm),
             policy::move)
        .def("maxPool",
             py::overload_cast<Tensor, Tensor, int, int, int, int, int, int,
                               int, int>(&Handler::maxPool),
             policy::move)
        .def("avgPool",
             py::overload_cast<Tensor, Tensor, int, int, int, int, int, int,
                               int, int>(&Handler::avgPool),
             policy::move)
        .def("add", py::overload_cast<Tensor, Tensor, Tensor>(&Handler::add),
             policy::move)
        .def("sub", py::overload_cast<Tensor, Tensor, Tensor>(&Handler::sub),
             policy::move)
        .def("mul", py::overload_cast<Tensor, Tensor, Tensor>(&Handler::mul),
             policy::move)
        .def("div", py::overload_cast<Tensor, Tensor, Tensor>(&Handler::div),
             policy::move)
        .def("pow", py::overload_cast<Tensor, Tensor, Tensor>(&Handler::pow),
             policy::move)
        .def("relu", py::overload_cast<Tensor, Tensor>(&Handler::relu),
             policy::move)
        .def("sigmoid", py::overload_cast<Tensor, Tensor>(&Handler::sigmoid),
             policy::move)
        .def("tanh", py::overload_cast<Tensor, Tensor>(&Handler::tanh),
             policy::move)
        .def("softmax", py::overload_cast<Tensor, Tensor>(&Handler::softmax),
             policy::move)
        .def("abs", py::overload_cast<Tensor, Tensor>(&Handler::abs),
             policy::move)
        .def("identity", py::overload_cast<Tensor, Tensor>(&Handler::identity),
             policy::move)
        .def("flatten", py::overload_cast<Tensor, Tensor>(&Handler::flatten),
             policy::move)
        .def("reshape",
             py::overload_cast<Tensor, Tensor, Shape>(&Handler::reshape),
             policy::move)
        .def("concat",
             py::overload_cast<TensorVec, Tensor, int>(&Handler::concat),
             policy::move)
        .def("gather",
             py::overload_cast<Tensor, Tensor, Tensor, int>(&Handler::gather),
             policy::move)
        .def("reduceMean",
             py::overload_cast<Tensor, Tensor, const optional<vector<int>> &,
                               bool>(&Handler::reduceMean),
             policy::move)
        .def("slice",
             py::overload_cast<
                 Tensor, Tensor, const vector<int> &, const vector<int> &,
                 const optional<vector<int>> &, const optional<vector<int>> &>(
                 &Handler::slice),
             policy::move)
        .def("pad",
             py::overload_cast<Tensor, Tensor, const vector<int> &,
                               const optional<vector<int>> &>(&Handler::pad),
             policy::move)
        .def("topo_sort", &Handler::topo_sort, policy::automatic)
        .def("operators", &Handler::operators, policy::move)
        .def("data_malloc", &Handler::data_malloc, policy::automatic)
        .def("run", &Handler::run, policy::automatic);
}

} // namespace infini

PYBIND11_MODULE(backend, m) {
    infini::register_operator_timer(m);
    infini::init_values(m);
    infini::init_graph_builder(m);
}
