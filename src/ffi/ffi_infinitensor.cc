#include "core/graph_handler.h"
#include "core/mutator.h"
#include "core/search_engine.h"
#include "nnet/nmutator.h"
#include "operators/batch_norm.h"
#include "operators/concat.h"
#include "operators/conv.h"
#include "operators/gather.h"
#include "operators/pooling.h"
#include "operators/reduce_mean.h"
#include "operators/reshape.h"
#include <pybind11/stl.h>

#ifdef USE_CUDA
#include "cuda/cuda_runtime.h"
#include "cuda/operator_timer.h"
#endif
#ifdef USE_MKL
#include "mkl/operator_timer.h"
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

#ifdef USE_MKL
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
        .VALUE(OpType, PRelu)
        .export_values();

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

static std::tuple<int, int, int, int, int, int> conv_attrs_of(Operator op) {
    IT_ASSERT(op->getOpType() == OpType::Conv);
    auto conv = dynamic_cast<const ConvObj *>(op.get());
    return std::make_tuple(conv->getPh(), conv->getPw(), conv->getDh(),
                           conv->getDw(), conv->getSh(), conv->getSw());
}

static std::tuple<float, float, bool> batch_norm_attrs_of(Operator op) {
    IT_ASSERT(op->getOpType() == OpType::BatchNorm);
    auto batchnorm = dynamic_cast<const BatchNormObj *>(op.get());
    return std::make_tuple(batchnorm->getMomentum(), batchnorm->getEps(),
                           batchnorm->getTraining());
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

static int concat_axis_of(Operator op) {
    IT_ASSERT(op->getOpType() == OpType::Concat);
    return dynamic_cast<const ConcatObj *>(op.get())->getDim();
}

static int gather_axis_of(Operator op) {
    IT_ASSERT(op->getOpType() == OpType::Gather);
    return dynamic_cast<const GatherObj *>(op.get())->getAxis();
}

static vector<int> reduce_mean_axes_of(Operator op) {
    IT_ASSERT(op->getOpType() == OpType::ReduceMean);
    auto &set = dynamic_cast<const ReduceMeanObj *>(op.get())->getAxes();
    return vector(set.begin(), set.end());
}

static Shape reshape_shape_of(Operator op) {
    IT_ASSERT(op->getOpType() == OpType::Reshape);
    return dynamic_cast<const ReshapeObj *>(op.get())->getShape();
}

void export_functions(py::module &m) {
#define FUNCTION(NAME) def(#NAME, &NAME)
    m.def("cpu_runtime", &NativeCpuRuntimeObj::getInstance)
#ifdef USE_CUDA
        .FUNCTION(cuda_runtime)
#endif
        .FUNCTION(conv_attrs_of)
        .FUNCTION(batch_norm_attrs_of)
        .FUNCTION(pool_attrs_of)
        .FUNCTION(tensor_dtype)
        .FUNCTION(reshape_shape_of)
        .FUNCTION(concat_axis_of)
        .FUNCTION(gather_axis_of)
        .FUNCTION(reduce_mean_axes_of);
#undef FUNCTION
}

void init_graph_builder(py::module &m) {
    using Handler = GraphHandlerObj;

    py::class_<Object, Ref<Object>>(m, "_Object")
        .def("__str__", &Object::toString)
        .def("guid", &Object::getGuid);
    py::class_<RuntimeObj, std::shared_ptr<RuntimeObj>>(m, "Runtime");
    py::class_<NativeCpuRuntimeObj, std::shared_ptr<NativeCpuRuntimeObj>,
               RuntimeObj>(m, "CpuRuntime");
#ifdef USE_CUDA
    py::class_<CudaRuntimeObj, std::shared_ptr<CudaRuntimeObj>, RuntimeObj>(
        m, "CudaRuntime");
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
        .def("data_malloc", &TensorObj::dataMalloc);
    py::class_<OperatorObj, std::shared_ptr<OperatorObj>, Object>(m, "Operator")
        .def("op_type", &OperatorObj::getOpType, policy::automatic)
        .def("inputs", py::overload_cast<>(&OperatorObj::getInputs, py::const_),
             policy::reference)
        .def("outputs",
             py::overload_cast<>(&OperatorObj::getOutputs, py::const_),
             policy::reference);
    py::class_<Handler>(m, "GraphHandler")
        .def(py::init<Runtime>())
        .def("tensor", &Handler::tensor, policy::move)
        .def("conv", &Handler::conv, policy::move)
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
        .def("identity", &Handler::identity, policy::move)
        .def("flatten", &Handler::flatten, policy::move)
        .def("reshape", &Handler::reshape, policy::move)
        .def("concat", &Handler::concat, policy::move)
        .def("gather", &Handler::gather, policy::move)
        .def("reduce_mean", &Handler::reduceMean, policy::move)
        .def("slice", &Handler::slice, policy::move)
        .def("pad", &Handler::pad, policy::move)
        .def("topo_sort", &Handler::topo_sort, policy::automatic)
        .def("operators", &Handler::operators, policy::move)
        .def("data_malloc", &Handler::data_malloc, policy::automatic)
        .def("run", &Handler::run, policy::automatic);
    py::class_<Mutator, Ref<Mutator>>(m, "Mutator").def("run", &Mutator::run);
    py::enum_<NMutator::Mode>(m, "NMutatorMode")
        .value("RuleBased", NMutator::Mode::RuleBased);
    py::class_<NMutator, Ref<NMutator>, Mutator>(m, "NMutator")
        .def(py::init<NMutator::Mode>())
        .def(py::init<NMutator::Mode, vector<int>>())
        .def("run", &NMutator::run)
        .def_static("memboundToJson", &NMutator::memboundToJson);
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

Graph getInfoGAN(int batch, Runtime runtime, int nLayers);
vector<Tensor> runInfoGAN(int nLayers);
Graph getConvtransposedNHWC(Runtime runtime, Shape shape, int layerId);
Graph optimizeGraph(Graph g, Runtime runtime, bool tuning);
void export_test_model(py::module &m) {
    m.def("runInfoGAN", &runInfoGAN);
    m.def("getInfoGAN", &getInfoGAN);
    m.def("getConvtransposedNHWC", &getConvtransposedNHWC);
    m.def("optimizeGraph", &optimizeGraph, "graph"_a, "runtime"_a,
          "tuning"_a = false);
}

} // namespace infini

PYBIND11_MODULE(backend, m) {
    infini::register_operator_timer(m);
    infini::export_values(m);
    infini::export_functions(m);
    infini::init_graph_builder(m);
    infini::export_test_model(m);
}
