#include <pybind11/stl.h>
#ifdef USE_CUDA
#include "cuda/operator_timer.h"
#endif
#include "core/graph_builder.h"
namespace py = pybind11;

namespace infini {

using namespace py::literals;
using policy = py::return_value_policy;

void register_operator_timer(py::module &m) {
#ifdef USE_CUDA
    using namespace opTimer;
    m.def("getPerfConvCudnn", &getPerfConvCudnn);
    m.def("getPerfConvBiasActCudnn", &getPerfConvBiasActCudnn);
    m.def("getPerfConvTransposed2dCudnn", &getPerfConvTransposed2dCudnn);
    m.def("getPerfMatmulCublas", &getPerfMatmulCublas);
    m.def("getPerfMatmulCublas", &getPerfMatmulCublas);
#endif
}

void init_graph_builder(py::module &m) {
    py::class_<RuntimeObj, std::shared_ptr<RuntimeObj>>(m, "RuntimeObj");
    py::class_<CpuRuntimeObj, std::shared_ptr<CpuRuntimeObj>, RuntimeObj>(
        m, "CpuRuntimeObj")
        .def(py::init<>())
        .def("getInstance", py::overload_cast<>(&CpuRuntimeObj::getInstance),
             policy::reference_internal);
    py::class_<Shape>(m, "Shape");
    py::class_<TensorObj, std::shared_ptr<TensorObj>>(m, "TensorObj");
    py::class_<Tensor>(m, "Tensor");
    py::class_<TensorVec>(m, "TensorVec");
    py::class_<OperatorObj, std::shared_ptr<OperatorObj>>(m, "OperatorObj");
    py::class_<Operator>(m, "Operator");
    py::class_<ActType>(m, "ActType");
    py::class_<ConvObj, std::shared_ptr<ConvObj>, OperatorObj>(m, "ConvObj");
    py::class_<MatmulObj, std::shared_ptr<MatmulObj>, OperatorObj>(m,
                                                                   "MatmulObj");
    py::class_<ConvTransposed2dObj, std::shared_ptr<ConvTransposed2dObj>,
               OperatorObj>(m, "ConvTransposed2dObj");
    py::class_<G2BMMObj, std::shared_ptr<G2BMMObj>, OperatorObj>(m, "G2BMMObj");
    py::class_<GBMMObj, std::shared_ptr<GBMMObj>, OperatorObj>(m, "GBMMObj");
    py::class_<PadObj, std::shared_ptr<PadObj>, OperatorObj>(m, "PadObj");
    py::class_<SliceObj, std::shared_ptr<SliceObj>, OperatorObj>(m, "SliceObj");
    py::class_<ConcatObj, std::shared_ptr<ConcatObj>, OperatorObj>(m,
                                                                   "ConcatObj");
    py::class_<SplitObj, std::shared_ptr<SplitObj>, OperatorObj>(m, "SplitObj");
    py::class_<ExtendObj, std::shared_ptr<ExtendObj>, OperatorObj>(m,
                                                                   "ExtendObj");
    py::class_<MaxPoolObj, std::shared_ptr<MaxPoolObj>, OperatorObj>(
        m, "MaxPoolObj");
    py::class_<AvgPoolObj, std::shared_ptr<AvgPoolObj>, OperatorObj>(
        m, "AvgPoolObj");
    py::class_<AddObj, std::shared_ptr<AddObj>, OperatorObj>(m, "AddObj");
    py::class_<SubObj, std::shared_ptr<SubObj>, OperatorObj>(m, "SubObj");
    py::class_<MulObj, std::shared_ptr<MulObj>, OperatorObj>(m, "MulObj");
    py::class_<DivObj, std::shared_ptr<DivObj>, OperatorObj>(m, "DivObj");
    py::class_<PowObj, std::shared_ptr<PowObj>, OperatorObj>(m, "PowObj");
    py::class_<GatherObj, std::shared_ptr<GatherObj>, OperatorObj>(m,
                                                                   "GatherObj");
    py::class_<ReshapeObj, std::shared_ptr<ReshapeObj>, OperatorObj>(
        m, "ReshapeObj");
    py::class_<FlattenObj, std::shared_ptr<FlattenObj>, OperatorObj>(
        m, "FlattenObj");
    py::class_<IdentityObj, std::shared_ptr<IdentityObj>, OperatorObj>(
        m, "IdentityObj");
    py::class_<SoftmaxObj, std::shared_ptr<SoftmaxObj>, OperatorObj>(
        m, "SoftmaxObj");
    py::class_<ReluObj, std::shared_ptr<ReluObj>, OperatorObj>(m, "ReluObj");
    py::class_<SigmoidObj, std::shared_ptr<SigmoidObj>, OperatorObj>(
        m, "SigmoidObj");
    py::class_<TanhObj, std::shared_ptr<TanhObj>, OperatorObj>(m, "TanhObj");
    py::class_<AbsObj, std::shared_ptr<AbsObj>, OperatorObj>(m, "AbsObj");
    py::class_<MemBoundObj, std::shared_ptr<MemBoundObj>, OperatorObj>(
        m, "MemBoundObj");
    py::class_<GraphBuilder>(m, "GraphBuilder");
    py::class_<GraphBuilderObj>(m, "GraphBuilderObj")
        .def(py::init<Runtime>())
        .def("tensor",
             py::overload_cast<Shape, const std::string &>(
                 &GraphBuilderObj::tensor),
             policy::reference_internal)
        .def("conv",
             py::overload_cast<Tensor, Tensor, Tensor, int, int, int, int, int,
                               int, Tensor>(&GraphBuilderObj::conv),
             policy::reference_internal)
        .def("matmul",
             py::overload_cast<Tensor, Tensor, Tensor, bool, bool, Tensor,
                               ActType>(&GraphBuilderObj::matmul),
             policy::reference_internal)
        .def("convTrans",
             py::overload_cast<Tensor, Tensor, Tensor, int, int, int, int, int,
                               int, int, int, int, Tensor, ActType>(
                 &GraphBuilderObj::convTrans),
             policy::reference_internal)
        .def("g2bmm",
             py::overload_cast<Tensor, Tensor, Tensor, const int, const int,
                               Tensor, ActType>(&GraphBuilderObj::g2bmm),
             policy::reference_internal)
        .def("gbmml",
             py::overload_cast<Tensor, Tensor, Tensor, const int, Tensor,
                               ActType>(&GraphBuilderObj::gbmml),
             policy::reference_internal)

        .def("pad",
             py::overload_cast<Tensor, Tensor, const vector<int> &,
                               const optional<const vector<int>> &>(
                 &GraphBuilderObj::pad),
             policy::reference_internal)
        .def("slice",
             py::overload_cast<Tensor, Tensor, const vector<int> &,
                               const vector<int> &,
                               const optional<const vector<int>> &,
                               const optional<const vector<int>> &>(
                 &GraphBuilderObj::slice),
             policy::reference_internal)
        .def(
            "concat",
            py::overload_cast<TensorVec, Tensor, int>(&GraphBuilderObj::concat),
            policy::reference_internal)
        .def("split",
             py::overload_cast<Tensor, std::optional<TensorVec>, int, int>(
                 &GraphBuilderObj::split),
             policy::reference_internal)
        .def("extend",
             py::overload_cast<Tensor, Tensor, int, int>(
                 &GraphBuilderObj::extend),
             policy::reference_internal)
        .def("maxpool",
             py::overload_cast<Tensor, Tensor, int, int, int, int, int, int,
                               int, int>(&GraphBuilderObj::maxpool),
             policy::reference_internal)
        .def("avgpool",
             py::overload_cast<Tensor, Tensor, int, int, int, int, int, int,
                               int, int>(&GraphBuilderObj::avgpool),
             policy::reference_internal)
        .def("add",
             py::overload_cast<Tensor, Tensor, Tensor>(&GraphBuilderObj::add),
             policy::reference_internal)
        .def("sub",
             py::overload_cast<Tensor, Tensor, Tensor>(&GraphBuilderObj::sub),
             policy::reference_internal)
        .def("mul",
             py::overload_cast<Tensor, Tensor, Tensor>(&GraphBuilderObj::mul),
             policy::reference_internal)
        .def("div",
             py::overload_cast<Tensor, Tensor, Tensor>(&GraphBuilderObj::div),
             policy::reference_internal)
        .def("pow",
             py::overload_cast<Tensor, Tensor, Tensor>(&GraphBuilderObj::pow),
             policy::reference_internal)
        .def("gather",
             py::overload_cast<Tensor, Tensor, Tensor, int>(
                 &GraphBuilderObj::gather),
             policy::reference_internal)
        .def("reshape",
             py::overload_cast<Tensor, Tensor, const Shape &>(
                 &GraphBuilderObj::reshape),
             policy::reference_internal)
        .def("flatten",
             py::overload_cast<Tensor, Tensor>(&GraphBuilderObj::flatten),
             policy::reference_internal)
        .def("identity",
             py::overload_cast<Tensor, Tensor>(&GraphBuilderObj::identity),
             policy::reference_internal)
        .def("softmax",
             py::overload_cast<Tensor, Tensor>(&GraphBuilderObj::softmax),
             policy::reference_internal)
        .def("relu", py::overload_cast<Tensor, Tensor>(&GraphBuilderObj::relu),
             policy::reference_internal)
        .def("sigmoid",
             py::overload_cast<Tensor, Tensor>(&GraphBuilderObj::sigmoid),
             policy::reference_internal)
        .def("tanh", py::overload_cast<Tensor, Tensor>(&GraphBuilderObj::tanh),
             policy::reference_internal)
        .def("abs", py::overload_cast<Tensor, Tensor>(&GraphBuilderObj::abs),
             policy::reference_internal)
        .def("memBound",
             py::overload_cast<const TensorVec &, const TensorVec &,
                               const std::vector<nnet::Tensor> &, nnet::Expr,
                               double, std::string>(&GraphBuilderObj::memBound),
             policy::reference_internal);
}

} // namespace infini

PYBIND11_MODULE(pyinfinitensor, m) {
    infini::register_operator_timer(m);
    infini::init_graph_builder(m);
}
