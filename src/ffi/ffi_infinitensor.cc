#include "core/graph_handler.h"
#include "core/mutator.h"
#include "core/search_engine.h"
#include "nnet/nmutator.h"
#include "nnet/test_models.h"
#include "operators/any.h"
#include "operators/batch_norm.h"
#include "operators/concat.h"
#include "operators/conv.h"
#include "operators/gather.h"
#include "operators/matmul.h"
#include "operators/membound.h"
#include "operators/pad.h"
#include "operators/pooling.h"
#include "operators/reduce_mean.h"
#include "operators/reshape.h"
#include "operators/split.h"
#include "operators/transpose.h"
#include "operators/unary.h"
#include <algorithm>
#include <pybind11/stl.h>

#ifdef USE_CUDA
#include "cuda/cuda_runtime.h"
#include "cuda/operator_timer.h"
#endif
#ifdef USE_BANG
#include "bang/bang_runtime.h"
#endif
#ifdef USE_INTELCPU
#include "intelcpu/mkl_runtime.h"
#include "intelcpu/operator_timer.h"
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

#ifdef USE_INTELCPU
    using namespace opTimer;
    m.def("getPerfConvMkl", &getPerfConvMkl);
    m.def("getPerfConvTransposed2dMkl", &getPerfConvTransposed2dMkl);
    m.def("getPerfMatmulMkl", &getPerfMatmulMkl);
#endif
}

void export_values(py::module &m) {
#define VALUE(TYPE, NAME) value(#NAME, TYPE::NAME)

    py::enum_<ActType>(m, "ActType")
        .value("Linear", ActType::None) // `None` is Python keyword
        .VALUE(ActType, Relu)
        .VALUE(ActType, Sigmoid)
        .VALUE(ActType, Tanh)
        .export_values();

    py::enum_<OpType>(m, "OpType")
        .VALUE(OpType, Unknown)
        .VALUE(OpType, Conv)
        .VALUE(OpType, Matmul)
        .VALUE(OpType, ConvTrans)
        .VALUE(OpType, ConvTransNHWC)
        .VALUE(OpType, ConvNHWC)
        .VALUE(OpType, G2BMM)
        .VALUE(OpType, GBMM)
        .VALUE(OpType, Pad)
        .VALUE(OpType, Clip)
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
        .VALUE(OpType, PRelu)
        .VALUE(OpType, Sigmoid)
        .VALUE(OpType, Tanh)
        .VALUE(OpType, Abs)
        .VALUE(OpType, Resize)
        .VALUE(OpType, Dropout)
        .VALUE(OpType, Conv2dReduce)
        .VALUE(OpType, Conv2dReduceTranspose)
        .VALUE(OpType, MemBound)
        .VALUE(OpType, Any)
        .export_values();

    py::enum_<TensorType>(m, "TensorType")
        .VALUE(TensorType, Input)
        .VALUE(TensorType, Initialized)
        .VALUE(TensorType, Other);
#undef VALUE
}

static int tensor_dtype(Tensor t) {
    if (t->getDType() == DataType::Float32)
        return OnnxDType::FLOAT;
    if (t->getDType() == DataType::UInt32)
        return OnnxDType::UINT32;
    if (t->getDType() == DataType::UInt8)
        return OnnxDType::UINT8;
    if (t->getDType() == DataType::Int8)
        return OnnxDType::INT8;
    if (t->getDType() == DataType::UInt16)
        return OnnxDType::UINT16;
    if (t->getDType() == DataType::Int16)
        return OnnxDType::INT16;
    if (t->getDType() == DataType::Int32)
        return OnnxDType::INT32;
    if (t->getDType() == DataType::Int64)
        return OnnxDType::INT64;
    IT_ASSERT(false, "Unsupported data type");
}

#ifdef USE_CUDA
static Ref<CudaRuntimeObj> cuda_runtime() { return make_ref<CudaRuntimeObj>(); }
#endif

#ifdef USE_BANG
static Ref<BangRuntimeObj> bang_runtime() { return make_ref<BangRuntimeObj>(); }
#endif

#ifdef USE_INTELCPU
static Ref<RuntimeObj> intelcpu_runtime() { return make_ref<MklRuntimeObj>(); }
#endif

static std::tuple<int, int, int, int, int, int> conv_attrs_of(Operator op) {
    IT_ASSERT(op->getOpType() == OpType::Conv ||
              op->getOpType() == OpType::ConvNHWC);
    auto conv = dynamic_cast<const ConvBaseObj *>(op.get());
    return std::make_tuple(conv->getPh(), conv->getPw(), conv->getSh(),
                           conv->getSw(), conv->getDh(), conv->getDw());
}

static std::tuple<int, int, int, int, int, int, int, int>
conv_trans_attrs_of(Operator op) {
    IT_ASSERT(op->getOpType() == OpType::ConvTrans ||
              op->getOpType() == OpType::ConvTransNHWC);
    auto conv = dynamic_cast<const ConvBaseObj *>(op.get());
    int oph, opw;

    if (op->getOpType() == OpType::ConvTrans) {
        auto _conv = dynamic_cast<const ConvTransposed2dObj *>(op.get());
        auto output_pad = _conv->getOutputPadding();
        oph = output_pad.first;
        opw = output_pad.second;
    } else {
        auto _conv = dynamic_cast<const ConvTransposed2dNHWCObj *>(op.get());
        auto output_pad = _conv->getOutputPadding();
        oph = output_pad.first;
        opw = output_pad.second;
    }

    return std::make_tuple(conv->getPh(), conv->getPw(), conv->getSh(),
                           conv->getSw(), conv->getDh(), conv->getDw(), oph,
                           opw);
}

static std::tuple<bool, bool> matmul_attrs_of(Operator op) {
    IT_ASSERT(op->getOpType() == OpType::Matmul);
    auto matmul = dynamic_cast<const MatmulObj *>(op.get());
    return std::make_tuple(matmul->getTransA(), matmul->getTransB());
}

static std::tuple<float, float, bool> batch_norm_attrs_of(Operator op) {
    IT_ASSERT(op->getOpType() == OpType::BatchNorm);
    auto batchnorm = dynamic_cast<const BatchNormObj *>(op.get());
    return std::make_tuple(batchnorm->getMomentum(), batchnorm->getEps(),
                           batchnorm->getTrainingMode());
}

static std::tuple<int, int, int, int, int, int, int, int>
pool_attrs_of(Operator op) {
    IT_ASSERT(op->getOpType() == OpType::MaxPool ||
              op->getOpType() == OpType::AvgPool);
    auto pool = dynamic_cast<const PoolingObj *>(op.get());
    return std::make_tuple(pool->getKh(), pool->getKw(), pool->getDh(),
                           pool->getDw(), pool->getPh(), pool->getPw(),
                           pool->getSh(), pool->getSw());
}

static std::tuple<std::optional<float>, std::optional<float>>
clip_attrs_of(Operator op) {
    IT_ASSERT(op->getOpType() == OpType::Clip);
    auto clip = dynamic_cast<const ClipObj *>(op.get());
    return std::make_tuple(clip->getMin(), clip->getMax());
}

static std::tuple<vector<int>, bool> reduce_mean_attrs_of(Operator op) {
    IT_ASSERT(op->getOpType() == OpType::ReduceMean);
    auto reduce_mean = dynamic_cast<const ReduceMeanObj *>(op.get());
    auto &set = reduce_mean->getAxes();
    return std::make_tuple(vector(set.begin(), set.end()),
                           reduce_mean->getKeepDims());
}

static int concat_axis_of(Operator op) {
    IT_ASSERT(op->getOpType() == OpType::Concat);
    return dynamic_cast<const ConcatObj *>(op.get())->getDim();
}

static int split_axis_of(Operator op) {
    IT_ASSERT(op->getOpType() == OpType::Split);
    return dynamic_cast<const SplitObj *>(op.get())->getDim();
}

static int gather_axis_of(Operator op) {
    IT_ASSERT(op->getOpType() == OpType::Gather);
    return dynamic_cast<const GatherObj *>(op.get())->getAxis();
}

static vector<int64_t> reshape_shape_of(Operator op) {
    IT_ASSERT(op->getOpType() == OpType::Reshape);
    auto shape = dynamic_cast<const ReshapeObj *>(op.get())->getShape();
    vector<int64_t> ans(shape.size());
    std::transform(shape.begin(), shape.end(), ans.begin(),
                   [](auto x) { return static_cast<int64_t>(x); });
    return ans;
}

static int flatten_axis_of(Operator op) {
    IT_ASSERT(op->getOpType() == OpType::Flatten);
    return as<FlattenObj>(op)->getAxis();
}

static vector<int64_t> pad_pads_of(Operator op) {
    IT_ASSERT(op->getOpType() == OpType::Pad);
    auto shape = dynamic_cast<const PadObj *>(op.get())->getPads();
    vector<int64_t> ans(shape.size());
    std::transform(shape.begin(), shape.end(), ans.begin(),
                   [](auto x) { return static_cast<int64_t>(x); });
    return ans;
}

static string any_kernelName_of(Operator op) {
    IT_ASSERT(op->getOpType() == OpType::Any);
    return as<AnyObj>(op)->getKernelName();
}

static vector<int> transpose_permute_of(Operator op) {
    IT_ASSERT(op->getOpType() == OpType::Transpose);
    return dynamic_cast<const TransposeObj *>(op.get())->getPermute();
}

static string membound_expr_of(Operator op) {
    return as<MemBoundObj>(op)->toJson();
}

void export_functions(py::module &m) {
#define FUNCTION(NAME) def(#NAME, &NAME)
    m.def("cpu_runtime", &NativeCpuRuntimeObj::getInstance)
#ifdef USE_CUDA
        .def("cuda_runtime", cuda_runtime)
#endif
#ifdef USE_INTELCPU
        .def("intelcpu_runtime", intelcpu_runtime)
#endif
#ifdef USE_CUDA
        .FUNCTION(cuda_runtime)
#endif
#ifdef USE_BANG
        .FUNCTION(bang_runtime)
#endif
        .FUNCTION(conv_attrs_of)
        .FUNCTION(conv_trans_attrs_of)
        .FUNCTION(matmul_attrs_of)
        .FUNCTION(batch_norm_attrs_of)
        .FUNCTION(pool_attrs_of)
        .FUNCTION(clip_attrs_of)
        .FUNCTION(reduce_mean_attrs_of)
        .FUNCTION(tensor_dtype)
        .FUNCTION(reshape_shape_of)
        .FUNCTION(flatten_axis_of)
        .FUNCTION(pad_pads_of)
        .FUNCTION(transpose_permute_of)
        .FUNCTION(concat_axis_of)
        .FUNCTION(split_axis_of)
        .FUNCTION(gather_axis_of)
        .FUNCTION(membound_expr_of)
        .FUNCTION(any_kernelName_of)
        .def("membound_hash_of",
             [](Operator op) { return as<MemBoundObj>(op)->getHash(); });
#undef FUNCTION
}

void init_graph_builder(py::module &m) {
    using Handler = GraphHandlerObj;

    py::class_<Object, Ref<Object>>(m, "_Object")
        .def("__str__", &Object::toString)
        .def("guid", &Object::getGuid);
    py::class_<RuntimeObj, Ref<RuntimeObj>>(m, "Runtime")
        .def("run", &RuntimeObj::run, "graph"_a, "tune"_a = false,
             "profiling"_a = false)
        .def("getPerfTime", &RuntimeObj::getPerfTime, "graph"_a, "profiling"_a,
             "allowEstimation"_a, "ignoreMemboundOp"_a)
        .def("timeNonCtcOperators", &RuntimeObj::timeNonCtcOperators);
    py::class_<NativeCpuRuntimeObj, std::shared_ptr<NativeCpuRuntimeObj>,
               RuntimeObj>(m, "CpuRuntime");
#ifdef USE_CUDA
    py::class_<CudaRuntimeObj, Ref<CudaRuntimeObj>, RuntimeObj>(m,
                                                                "CudaRuntime")
        .def("timeWithCudaGraph",
             py::overload_cast<Graph, int>(&CudaRuntimeObj::timeWithCudaGraph))
        .def("setEnableTF32", &CudaRuntimeObj::setEnableTF32);
#endif
#ifdef USE_BANG
    py::class_<BangRuntimeObj, std::shared_ptr<BangRuntimeObj>, RuntimeObj>(
        m, "BangRuntime");
#endif
    py::class_<TensorObj, std::shared_ptr<TensorObj>, Object>(m, "Tensor")
        .def("fuid", &TensorObj::getFuid, policy::automatic)
        .def("shape", &TensorObj::getDims, policy::move)
        .def("copyin_float", &TensorObj::copyin<float>, policy::move)
        .def("copyin_int32", &TensorObj::copyin<int32_t>, policy::move)
        .def("copyin_int64", &TensorObj::copyin<int64_t>, policy::move)
        .def("copyout_float", &TensorObj::copyout<float>, policy::move)
        .def("copyout_int32", &TensorObj::copyout<int32_t>, policy::move)
        .def("copyout_int64", &TensorObj::copyout<int64_t>, policy::move)
        .def("has_target", &TensorObj::hasTarget, policy::automatic)
        .def("src", &TensorObj::getSource, policy::move)
        .def("print_data", &TensorObj::printData)
        .def("data_malloc", &TensorObj::dataMalloc)
        .def("getTensorType", &TensorObj::getTensorType);
    py::class_<OperatorObj, std::shared_ptr<OperatorObj>, Object>(m, "Operator")
        .def("op_type", &OperatorObj::getOpType, policy::automatic)
        .def("inputs", py::overload_cast<>(&OperatorObj::getInputs, py::const_),
             policy::reference)
        .def("outputs",
             py::overload_cast<>(&OperatorObj::getOutputs, py::const_),
             policy::reference);
    py::class_<Handler>(m, "GraphHandler")
        .def(py::init<Runtime>())
        .def(py::init<Graph>())
        .def("inputs", &Handler::inputs, policy::move)
        .def("outputs", &Handler::outputs, policy::move)
        .def("tensor", &Handler::tensor, policy::move, "shape"_a, "dtype"_a = 1,
             "tensor_type"_a = TensorType::Other)
        .def("conv", &Handler::conv, policy::move)
        .def("convTransposed2d", &Handler::convTransposed2d, policy::move)
        .def("convNHWC", &Handler::convNHWC, policy::move)
        .def("convtransposed2dNHWC", &Handler::convTransposed2dNHWC,
             policy::move)
        .def("matmul", &Handler::matmul, policy::move)
        .def("batchNorm", &Handler::batchNorm, policy::move)
        .def("maxPool", &Handler::maxPool, policy::move)
        .def("avgPool", &Handler::avgPool, policy::move)
        .def("add", &Handler::add, policy::move)
        .def("sub", &Handler::sub, policy::move)
        .def("mul", &Handler::mul, policy::move)
        .def("div", &Handler::div, policy::move)
        .def("pow", &Handler::pow, policy::move)
        .def("relu", &Handler::relu, policy::move)
        .def("sigmoid", &Handler::sigmoid, policy::move)
        .def("tanh", &Handler::tanh, policy::move)
        .def("softmax", &Handler::softmax, policy::move)
        .def("abs", &Handler::abs, policy::move)
        .def("shape", &Handler::shape, policy::move)
        .def("identity", &Handler::identity, policy::move)
        .def("flatten", &Handler::flatten, policy::move)
        .def("pRelu", &Handler::pRelu, policy::move)
        .def("clip", &Handler::clip, policy::move)
        .def("transpose", &Handler::transpose, policy::move)
        .def("reshape", &Handler::reshape, policy::move)
        .def("concat", &Handler::concat, policy::move)
        .def("split", &Handler::split, policy::move)
        .def("gather", &Handler::gather, policy::move)
        .def("reduce_mean", &Handler::reduceMean, policy::move)
        .def("slice", &Handler::slice, policy::move)
        .def("pad", &Handler::pad, policy::move)
        .def("memBound", &Handler::memBound, policy::move)
        .def("topo_sort", &Handler::topo_sort, policy::automatic)
        .def("optimize", &Handler::optimize, policy::automatic)
        .def("operators", &Handler::operators, policy::move)
        .def("data_malloc", &Handler::data_malloc, policy::automatic)
        .def("run", &Handler::run, policy::automatic)
        .def("getGraph", &Handler::getGraph);
    py::class_<Mutator, Ref<Mutator>>(m, "Mutator").def("run", &Mutator::run);
    py::enum_<NMutator::Mode>(m, "NMutatorMode")
        .value("Normal", NMutator::Mode::Normal)
        .value("RuleBased", NMutator::Mode::RuleBased);
    py::class_<NMutator, Ref<NMutator>, Mutator>(m, "NMutator")
        .def(py::init<NMutator::Mode>())
        .def(py::init<NMutator::Mode, vector<int>>())
        .def("run", &NMutator::run);
    py::class_<SearchEngine>(m, "SearchEngine")
        .def(py::init<Runtime, Ref<Mutator>>())
        .def("run", &SearchEngine::run);
    py::class_<GraphObj, Ref<GraphObj>, Object>(m, "Graph")
        .def("tensors", &GraphObj::getTensors)
        .def("operators", &GraphObj::getOperators)
        .def("inputs", &GraphObj::getInputs)
        .def("outputs", &GraphObj::getOutputs)
        .def("print", &GraphObj::print)
        .def("topo_sort", &GraphObj::topo_sort);
}

void export_test_model(py::module &m) {
#ifdef USE_CUDA
    m.def("runInfoGAN", &runInfoGAN)
        .def("getGANGraph", &getGANGraph)
        .def("getFSRCNNGraph", &getFSRCNNGraph)
        .def("getLongformer", &getLongformer)
        .def("getConvtransposedNHWC", &getConvtransposedNHWC)
        .def("optimizeGraph", &optimizeGraph, "graph"_a, "runtime"_a,
             "tuning"_a = false, "mode"_a = NMutator::Mode::Normal,
             "rules"_a = vector<int>{})
        .def("initializeGraphTensors", &initializeGraphTensors, "g"_a,
             "l"_a = -0.1, "r"_a = 0.1, "useInt"_a = false)
        .def("convertNCHWtoNHWCModel", &convertNCHWtoNHWCModel)
        .def("optimizeWithDepthConstraint", &optimizeWithDepthConstraint)
        .def("optimizeModel", &optimizeModel);
#endif
}

} // namespace infini

PYBIND11_MODULE(backend, m) {
    infini::register_operator_timer(m);
    infini::export_values(m);
    infini::export_functions(m);
    infini::init_graph_builder(m);
    infini::export_test_model(m);
}
