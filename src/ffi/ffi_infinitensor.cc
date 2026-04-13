#include "core/data_type.h"
#include "core/graph_handler.h"
#include "operators/batch_norm.h"
#include "operators/concat.h"
#include "operators/conv.h"
#include "operators/expand.h"
#include "operators/gather.h"
#include "operators/lrn.h"
#include "operators/matmul.h"
#include "operators/pad.h"
#include "operators/pooling.h"
#include "operators/reduce.h"
#include "operators/reshape.h"
#include "operators/split.h"
#include "operators/squeeze.h"
#include "operators/transpose.h"
#include "operators/unary.h"
#include "operators/unsqueeze.h"
#include <algorithm>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#ifdef USE_CUDA
#include "cuda/cuda_runtime.h"
#include "cuda/operator_timer.h"
#endif
#ifdef USE_BANG
#include "bang/bang_runtime.h"
#endif
#ifdef USE_KUNLUN
#include "kunlun/kunlun_runtime.h"
#endif
#ifdef USE_ASCEND
#include "ascend/ascend_runtime.h"
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

decltype(OpType::type) getId(OpType const *const ptr) { return ptr->type; }

void export_values(py::module &m) {
#define VALUE(TYPE, NAME) value(#NAME, TYPE::NAME)

    py::enum_<ActType>(m, "ActType")
        .value("Linear", ActType::None) // `None` is Python keyword
        .VALUE(ActType, Relu)
        .VALUE(ActType, LeakyRelu)
        .VALUE(ActType, Sigmoid)
        .VALUE(ActType, Tanh)
        .export_values();

    py::class_<OpType>(m, "OpType")
        .def(py::init<decltype(OpType::type)>())
        .def("id", getId, policy::automatic);
    py::enum_<decltype(OpType::type)>(m, "OpTypeId")
        .VALUE(OpType, Conv)
        .VALUE(OpType, MatMul)
        .VALUE(OpType, ConvTranspose)
        .VALUE(OpType, Pad)
        .VALUE(OpType, Clip)
        .VALUE(OpType, Slice)
        .VALUE(OpType, Concat)
        .VALUE(OpType, Split)
        .VALUE(OpType, Transpose)
        .VALUE(OpType, Extend)
        .VALUE(OpType, MaxPool)
        .VALUE(OpType, AveragePool)
        .VALUE(OpType, Add)
        .VALUE(OpType, Sub)
        .VALUE(OpType, Mul)
        .VALUE(OpType, Div)
        .VALUE(OpType, Pow)
        .VALUE(OpType, Gather)
        .VALUE(OpType, GatherElements)
        .VALUE(OpType, ReduceMean)
        .VALUE(OpType, ReduceSum)
        .VALUE(OpType, Reshape)
        .VALUE(OpType, Squeeze)
        .VALUE(OpType, Unsqueeze)
        .VALUE(OpType, Flatten)
        .VALUE(OpType, Identity)
        .VALUE(OpType, BatchNormalization)
        .VALUE(OpType, Softmax)
        .VALUE(OpType, Relu)
        .VALUE(OpType, LeakyRelu)
        .VALUE(OpType, Gelu)
        .VALUE(OpType, PRelu)
        .VALUE(OpType, Sigmoid)
        .VALUE(OpType, Tanh)
        .VALUE(OpType, HardSigmoid)
        .VALUE(OpType, HardSwish)
        .VALUE(OpType, Abs)
        .VALUE(OpType, Resize)
        .VALUE(OpType, Dropout)
        .VALUE(OpType, Cast)
        .VALUE(OpType, Sqrt)
        .VALUE(OpType, Neg)
        .VALUE(OpType, Expand)
        .VALUE(OpType, Erf)
        .VALUE(OpType, Where)
        .VALUE(OpType, DepthToSpace)
        .VALUE(OpType, LRN)
        .VALUE(OpType, Elu)
        .VALUE(OpType, ArgMax)
        .export_values();

#undef VALUE
}

static int tensor_dtype(Tensor t) {
    if (t->getDType() == DataType::Undefine)
        return 0;
    if (t->getDType() == DataType::Float32)
        return 1;
    if (t->getDType() == DataType::UInt8)
        return 2;
    if (t->getDType() == DataType::Int8)
        return 3;
    if (t->getDType() == DataType::UInt16)
        return 4;
    if (t->getDType() == DataType::Int16)
        return 5;
    if (t->getDType() == DataType::Int32)
        return 6;
    if (t->getDType() == DataType::Int64)
        return 7;
    if (t->getDType() == DataType::String)
        return 8;
    if (t->getDType() == DataType::Bool)
        return 9;
    if (t->getDType() == DataType::Float16)
        return 10;
    if (t->getDType() == DataType::Double)
        return 11;
    if (t->getDType() == DataType::UInt32)
        return 12;
    if (t->getDType() == DataType::UInt64)
        return 13;
    if (t->getDType() == DataType::BFloat16)
        return 16;
    IT_ASSERT(false, "Unsupported data type");
}

#ifdef USE_CUDA
// NOTE(lizhouyang): deprecate this, use CudaRuntime directly.
[[deprecated]] static Ref<CudaRuntimeObj> cuda_runtime() {
    return make_ref<CudaRuntimeObj>(0);
}
#endif

#ifdef USE_BANG
static Ref<BangRuntimeObj> bang_runtime() { return make_ref<BangRuntimeObj>(); }
#endif

#ifdef USE_KUNLUN
static Ref<KUNLUNRuntimeObj> kunlun_runtime() {
    return make_ref<KUNLUNRuntimeObj>();
}
#endif

#ifdef USE_ASCEND
static Ref<ASCENDRuntimeObj> ascend_runtime() {
    return make_ref<ASCENDRuntimeObj>();
}
#endif

#ifdef USE_INTELCPU
static Ref<RuntimeObj> intelcpu_runtime() { return make_ref<MklRuntimeObj>(); }
#endif

static std::tuple<int, int, int, int, int, int> conv_attrs_of(Operator op) {
    IT_ASSERT(op->getOpType() == OpType::Conv);
    auto conv = dynamic_cast<const ConvObj *>(op.get());
    return std::make_tuple(conv->getPh(), conv->getPw(), conv->getDh(),
                           conv->getDw(), conv->getSh(), conv->getSw());
}

static std::tuple<int, int, int, int, int, int, int, int>
conv_trans_attrs_of(Operator op) {
    IT_ASSERT(op->getOpType() == OpType::ConvTranspose);
    auto conv = dynamic_cast<const ConvTransposed2dObj *>(op.get());
    auto [oph, opw] = conv->getOutputPadding();
    return std::make_tuple(conv->getPh(), conv->getPw(), conv->getDh(),
                           conv->getDw(), conv->getSh(), conv->getSw(), oph,
                           opw);
}

static std::tuple<bool, bool> matmul_attrs_of(Operator op) {
    IT_ASSERT(op->getOpType() == OpType::MatMul);
    auto matmul = dynamic_cast<const MatmulObj *>(op.get());
    return std::make_tuple(matmul->getTransA(), matmul->getTransB());
}

static float elu_alpha_of(Operator op) {
    IT_ASSERT(op->getOpType() == OpType::Elu);
    auto elu = dynamic_cast<const EluObj *>(op.get());
    return elu->getAlpha();
}

static std::tuple<float, float, bool> batch_norm_attrs_of(Operator op) {
    IT_ASSERT(op->getOpType() == OpType::BatchNormalization);
    auto batchnorm = dynamic_cast<const BatchNormObj *>(op.get());
    return std::make_tuple(batchnorm->getMomentum(), batchnorm->getEps(),
                           batchnorm->getTrainingMode());
}

static std::tuple<int, int, int, int, int, int, int, int, int>
pool_attrs_of(Operator op) {
    IT_ASSERT(op->getOpType() == OpType::MaxPool ||
              op->getOpType() == OpType::AveragePool);
    auto pool = dynamic_cast<const PoolingObj *>(op.get());
    return std::make_tuple(pool->getKh(), pool->getKw(), pool->getDh(),
                           pool->getDw(), pool->getPh(), pool->getPw(),
                           pool->getSh(), pool->getSw(), pool->getCeilMode());
}

static std::tuple<std::optional<float>, std::optional<float>>
clip_attrs_of(Operator op) {
    IT_ASSERT(op->getOpType() == OpType::Clip);
    auto clip = dynamic_cast<const ClipObj *>(op.get());
    return std::make_tuple(clip->getMin(), clip->getMax());
}

static std::tuple<vector<int>, bool> reduce_attrs_of(Operator op) {
    IT_ASSERT(op->getOpType() == OpType::ReduceMean ||
              op->getOpType() == OpType::ReduceSum);
    auto reduce = dynamic_cast<const ReduceBaseObj *>(op.get());
    auto &set = reduce->getAxes();
    return std::make_tuple(vector(set.begin(), set.end()),
                           reduce->getKeepDims());
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
    IT_ASSERT(op->getOpType() == OpType::Gather ||
              op->getOpType() == OpType::GatherElements);
    return dynamic_cast<const GatherBaseObj *>(op.get())->getAxis();
}

static vector<int64_t> reshape_shape_of(Operator op) {
    IT_ASSERT(op->getOpType() == OpType::Reshape);
    auto shape = dynamic_cast<const ReshapeObj *>(op.get())->getShape();
    vector<int64_t> ans(shape.size());
    std::transform(shape.begin(), shape.end(), ans.begin(),
                   [](auto x) { return static_cast<int64_t>(x); });
    return ans;
}

static vector<int64_t> squeeze_axes_of(Operator op) {
    IT_ASSERT(op->getOpType() == OpType::Squeeze);
    auto axes = dynamic_cast<const SqueezeObj *>(op.get())->getAxes();
    vector<int64_t> ans(axes.size());
    std::transform(axes.begin(), axes.end(), ans.begin(),
                   [](auto x) { return static_cast<int64_t>(x); });
    return ans;
}

static vector<int64_t> unsqueeze_axes_of(Operator op) {
    IT_ASSERT(op->getOpType() == OpType::Unsqueeze);
    auto axes = dynamic_cast<const UnsqueezeObj *>(op.get())->getAxes();
    vector<int64_t> ans(axes.size());
    std::transform(axes.begin(), axes.end(), ans.begin(),
                   [](auto x) { return static_cast<int64_t>(x); });
    return ans;
}

static vector<int64_t> expand_shape_of(Operator op) {
    IT_ASSERT(op->getOpType() == OpType::Expand);
    auto shape = dynamic_cast<const ExpandObj *>(op.get())->getShape();
    vector<int64_t> ans(shape.size());
    std::transform(shape.begin(), shape.end(), ans.begin(),
                   [](auto x) { return static_cast<int64_t>(x); });
    return ans;
}

static vector<int64_t> pad_pads_of(Operator op) {
    IT_ASSERT(op->getOpType() == OpType::Pad);
    auto shape = dynamic_cast<const PadObj *>(op.get())->getPads();
    vector<int64_t> ans(shape.size());
    std::transform(shape.begin(), shape.end(), ans.begin(),
                   [](auto x) { return static_cast<int64_t>(x); });
    return ans;
}

static vector<int> transpose_permute_of(Operator op) {
    IT_ASSERT(op->getOpType() == OpType::Transpose);
    return dynamic_cast<const TransposeObj *>(op.get())->getPermute();
}

static int flatten_axis_of(Operator op) {
    IT_ASSERT(op->getOpType() == OpType::Flatten);
    return dynamic_cast<const FlattenObj *>(op.get())->getAxis();
}

static int cast_to_of(Operator op) {
    IT_ASSERT(op->getOpType() == OpType::Cast);
    auto castOutputDtype =
        dynamic_cast<const CastObj *>(op.get())->getOutputDataType();
    return castOutputDtype.getIndex();
}

static std::tuple<int, std::string> depth_to_space_attrs_of(Operator op) {
    IT_ASSERT(op->getOpType() == OpType::DepthToSpace);
    auto depth_to_space = dynamic_cast<const DepthToSpaceObj *>(op.get());
    return std::make_tuple(depth_to_space->getBlockSize(),
                           depth_to_space->getModeString());
}

static std::tuple<float, float, float, int> lrn_attrs_of(Operator op) {
    IT_ASSERT(op->getOpType() == OpType::LRN);
    auto lrn = dynamic_cast<const LRNObj *>(op.get());
    auto [alpha, beta, bias] = lrn->getAlphaBetaBias();
    auto size = lrn->getSize();
    return std::make_tuple(alpha, beta, bias, size);
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

#ifdef USE_KUNLUN
        .FUNCTION(kunlun_runtime)
#endif

#ifdef USE_ASCEND
        .FUNCTION(ascend_runtime)
#endif
        .FUNCTION(conv_attrs_of)
        .FUNCTION(conv_trans_attrs_of)
        .FUNCTION(matmul_attrs_of)
        .FUNCTION(batch_norm_attrs_of)
        .FUNCTION(pool_attrs_of)
        .FUNCTION(clip_attrs_of)
        .FUNCTION(reduce_attrs_of)
        .FUNCTION(tensor_dtype)
        .FUNCTION(reshape_shape_of)
        .FUNCTION(expand_shape_of)
        .FUNCTION(pad_pads_of)
        .FUNCTION(transpose_permute_of)
        .FUNCTION(concat_axis_of)
        .FUNCTION(split_axis_of)
        .FUNCTION(gather_axis_of)
        .FUNCTION(flatten_axis_of)
        .FUNCTION(cast_to_of)
        .FUNCTION(depth_to_space_attrs_of)
        .FUNCTION(squeeze_axes_of)
        .FUNCTION(unsqueeze_axes_of)
        .FUNCTION(lrn_attrs_of)
        .FUNCTION(elu_alpha_of);
#undef FUNCTION
}

// A helper function that converts DataType to python format string
static std::string getFormat(DataType type) {
    std::string format;
    if (type == DataType::Float32) {
        format = py::format_descriptor<float>::format();
    } else if (type == DataType::Double) {
        format = py::format_descriptor<double>::format();
    } else if (type == DataType::Int32) {
        format = py::format_descriptor<int>::format();
    } else if (type == DataType::UInt32) {
        format = py::format_descriptor<uint32_t>::format();
    } else if (type == DataType::Int64) {
        format = py::format_descriptor<int64_t>::format();
    } else if (type == DataType::UInt64) {
        format = py::format_descriptor<uint64_t>::format();
    } else if (type == DataType::Int16) {
        format = py::format_descriptor<int16_t>::format();
    } else if (type == DataType::UInt16) {
        format = py::format_descriptor<uint16_t>::format();
    } else if (type == DataType::Int8) {
        format = py::format_descriptor<int8_t>::format();
    } else if (type == DataType::UInt8) {
        format = py::format_descriptor<uint8_t>::format();
    } else if (type == DataType::Bool) {
        format = py::format_descriptor<bool>::format();
    } else if (type == DataType::Float16 || type == DataType::BFloat16) {
        // Python uses "e" for half precision float type code.
        // Check the following link for more information.
        // https://docs.python.org/3/library/struct.html#format-characters
        format = "e";
    } else {
        throw std::runtime_error("Error converting TensorObj to "
                                 "Numpy: unsupported datatype.\n");
    }

    return format;
}

void init_graph_builder(py::module &m) {
    using Handler = GraphHandlerObj;

    py::class_<RuntimeObj, std::shared_ptr<RuntimeObj>>(m, "Runtime");
    py::class_<NativeCpuRuntimeObj, std::shared_ptr<NativeCpuRuntimeObj>,
               RuntimeObj>(m, "CpuRuntime");
#ifdef USE_CUDA
    py::class_<CudaRuntimeObj, std::shared_ptr<CudaRuntimeObj>, RuntimeObj>(
        m, "CudaRuntime")
        .def(py::init<int>(), py::arg("device") = 0)
        .def("init_comm", &CudaRuntimeObj::initComm);
#endif
#ifdef USE_BANG
    py::class_<BangRuntimeObj, std::shared_ptr<BangRuntimeObj>, RuntimeObj>(
        m, "BangRuntime")
        .def(py::init<int>(), py::arg("device") = 0)
        .def("init_comm", &BangRuntimeObj::initComm);
#endif
#ifdef USE_KUNLUN
    py::class_<KUNLUNRuntimeObj, std::shared_ptr<KUNLUNRuntimeObj>, RuntimeObj>(
        m, "KUNLUNRuntime")
        .def(py::init<int>(), py::arg("device") = 0)
        .def("init_comm", &KUNLUNRuntimeObj::initComm);
#endif

#ifdef USE_ASCEND
    py::class_<ASCENDRuntimeObj, std::shared_ptr<ASCENDRuntimeObj>, RuntimeObj>(
        m, "ASCENDRuntime")
        .def(py::init<int>(), py::arg("device") = 0)
        .def("init_comm", &ASCENDRuntimeObj::initComm);
    ;
#endif
    py::class_<TensorObj, std::shared_ptr<TensorObj>>(m, "Tensor",
                                                      py::buffer_protocol())
        .def("fuid", &TensorObj::getFuid, policy::automatic)
        .def("shape", &TensorObj::getDims, policy::move)
        .def("set_weight", &TensorObj::setWeight, policy::move)
        .def("set_input", &TensorObj::setInput, policy::move)
        .def("set_output", &TensorObj::setOutput, policy::move)
        .def("dtype", &TensorObj::getDTypeIndex, policy::automatic)
        .def("copyin_float", &TensorObj::copyin<float>, policy::move)
        .def("copyin_int32", &TensorObj::copyin<int32_t>, policy::move)
        .def("copyin_int64", &TensorObj::copyin<int64_t>, policy::move)
        .def("copyin_int8", &TensorObj::copyin<int8_t>, policy::move)
        .def("copyin_uint8", &TensorObj::copyin<uint8_t>, policy::move)
        .def("copyin_float16", &TensorObj::copyin<uint16_t>, policy::move)
        .def("copyout_float", &TensorObj::copyout<float>, policy::move)
        .def("copyout_int32", &TensorObj::copyout<int32_t>, policy::move)
        .def("copyout_int64", &TensorObj::copyout<int64_t>, policy::move)
        .def("copyout_int8", &TensorObj::copyout<int8_t>, policy::move)
        .def("copyout_uint8", &TensorObj::copyout<uint8_t>, policy::move)
        .def("copyout_float16", &TensorObj::copyout<uint16_t>, policy::move)
        // Copy data from a Numpy array
        .def("copyin_numpy",
             [](TensorObj &self, py::buffer buf) {
                 py::buffer_info buf_info = buf.request();
                 void *data_np = buf_info.ptr;
                 size_t itemsize = buf_info.itemsize;
                 size_t size = buf_info.size;
                 std::cout << "Numpy array itemsize: " << itemsize
                           << ", size: " << size << std::endl;
                 std::cout << "Tensor dtype size: "
                           << self.getDType().getSize()
                           << ", Tensor size: " << self.size() << std::endl;
                 IT_ASSERT(itemsize == self.getDType().getSize());
                 IT_ASSERT(size == self.size());
                 for (size_t i = 0; i < self.getRank(); i++) {
                     IT_ASSERT(self.getDims()[i] == buf_info.shape[i]);
                 }
                 self.copyin(data_np, self.getBytes());
             })
        // Return a Numpy array which copies the values of this tensor
        .def("copyout_numpy",
             [](TensorObj &self) -> py::array {
                 vector<size_t> stride_byte;
                 for (int s : self.getStride()) {
                     stride_byte.push_back(s * self.getDType().getSize());
                 }
                 std::string format = getFormat(self.getDType());

                 py::array numpy_array(py::dtype(format), self.getDims(),
                                       nullptr);

                 // Copy data to the numpy array
                 auto ptr = numpy_array.mutable_data();
                 self.copyout(ptr, self.getBytes());

                 return numpy_array;
             })
        .def("has_target", &TensorObj::hasTarget, policy::automatic)
        .def("src", &TensorObj::getSource, policy::move)
        .def("printData", &TensorObj::printData, policy::automatic)
        .def("copy_data",
             py::overload_cast<const Tensor &>(&TensorObj::copyData),
             policy::move);
    py::class_<OperatorObj, std::shared_ptr<OperatorObj>>(m, "Operator")
        .def("op_type", &OperatorObj::getOpType, policy::automatic)
        .def("inputs", py::overload_cast<>(&OperatorObj::getInputs, py::const_),
             policy::reference)
        .def("outputs",
             py::overload_cast<>(&OperatorObj::getOutputs, py::const_),
             policy::reference);
    py::class_<Handler>(m, "GraphHandler")
        .def(py::init<Runtime>())
        .def("elu", &Handler::elu, policy::move)
        .def("tensor", &Handler::tensor, policy::move)
        .def("conv", &Handler::conv, policy::move)
        .def("convTransposed2d", &Handler::convTransposed2d, policy::move)
        .def("matmul", &Handler::matmul, policy::move)
        .def("batchNormalization", &Handler::batchNormalization, policy::move)
        .def("layerNormalization", &Handler::layerNormalization, policy::move)
        .def("instanceNormalization", &Handler::instanceNormalization,
             policy::move)
        .def("RMSNorm", &Handler::rmsNorm, policy::move)
        .def("maxPool", &Handler::maxPool, policy::move)
        .def("avgPool", &Handler::avgPool, policy::move)
        .def("add", &Handler::add, policy::move)
        .def("sub", &Handler::sub, policy::move)
        .def("mul", &Handler::mul, policy::move)
        .def("max", &Handler::max, policy::move)
        .def("div", &Handler::div, policy::move)
        .def("pow", &Handler::pow, policy::move)
        .def("min", &Handler::min, policy::move)
        .def("max", &Handler::max, policy::move)
        .def("relu", &Handler::relu, policy::move)
        .def("leakyRelu", &Handler::leakyRelu, policy::move)
        .def("silu", &Handler::silu, policy::move)
        .def("gelu", &Handler::gelu, policy::move)
        .def("sigmoid", &Handler::sigmoid, policy::move)
        .def("tanh", &Handler::tanh, policy::move)
        .def("hardSigmoid", &Handler::hardSigmoid, policy::move)
        .def("hardSwish", &Handler::hardSwish, policy::move)
        .def("softmax", &Handler::softmax, policy::move)
        .def("abs", &Handler::abs, policy::move)
        .def("sqrt", &Handler::sqrt, policy::move)
        .def("neg", &Handler::neg, policy::move)
        .def("shape", &Handler::shape, policy::move)
        .def("identity", &Handler::identity, policy::move)
        .def("flatten", &Handler::flatten, policy::move)
        .def("pRelu", &Handler::pRelu, policy::move)
        .def("clip", &Handler::clip, policy::move)
        .def("transpose", &Handler::transpose, policy::move)
        .def("depthToSpace", &Handler::depthToSpace, policy::move)
        .def("reshape", &Handler::reshape, policy::move)
        .def("resize", &Handler::resize, policy::move)
        .def("squeeze", &Handler::squeeze, policy::move)
        .def("unsqueeze", &Handler::unsqueeze, policy::move)
        .def("concat", &Handler::concat, policy::move)
        .def("attentionKVCache", &Handler::attentionKVCache, policy::move)
        .def("RoPE", &Handler::RoPE, policy::move)
        .def("split", &Handler::split, policy::move)
        .def("gather", &Handler::gather, policy::move)
        .def("gatherElements", &Handler::gatherElements, policy::move)
        .def("reduceMean", &Handler::reduceMean, policy::move)
        .def("reduceSum", &Handler::reduceSum, policy::move)
        .def("slice", &Handler::slice, policy::move)
        .def("pad", &Handler::pad, policy::move)
        .def("allReduceSum", &Handler::allReduceSum, policy::move)
        .def("allReduceProd", &Handler::allReduceProd, policy::move)
        .def("allReduceMin", &Handler::allReduceMin, policy::move)
        .def("allReduceMax", &Handler::allReduceMax, policy::move)
        .def("allReduceAvg", &Handler::allReduceAvg, policy::move)
        .def("allGather", &Handler::allGather, policy::move)
        .def("broadcast", &Handler::broadcast, policy::move)
        .def("send", &Handler::send, policy::move)
        .def("recv", &Handler::recv, policy::move)
        .def("cast", &Handler::cast, policy::move)
        .def("expand", &Handler::expand, policy::move)
        .def("erf", &Handler::erf, policy::move)
        .def("where", &Handler::where, policy::move)
        .def("lrn", &Handler::lrn, policy::move)
        .def("topo_sort", &Handler::topo_sort, policy::automatic)
        .def("optimize", &Handler::optimize, policy::automatic)
        .def("operators", &Handler::operators, policy::move)
        .def("data_malloc", &Handler::data_malloc,
             py::arg("useNaiveAllocator") = false, py::arg("memPoolSize") = 0,
             policy::automatic)
        .def("clone_KV", &Handler::clone_KV, policy::move)
        .def("free_heap", &Handler::free_heap, policy::move)
        .def("get_perf_time", &Handler::get_perf_time, policy::automatic)
        .def("tune", &Handler::tune, policy::automatic)
        .def("run", &Handler::run, policy::automatic)
        .def("argmax", &Handler::argmax, policy::move)

#ifdef USE_CUDA
        .def("run_with_cudagraph", &Handler::run_with_cudagraph,
             policy::automatic)
#endif
        .def("shape_infer", &Handler::shape_infer, policy::automatic)
        .def("change_shape", &Handler::change_shape, policy::automatic)
        .def("getDims", &Handler::getDims, policy::automatic)
        .def("get_perf_time", &Handler::get_perf_time, policy::automatic);
}

} // namespace infini

PYBIND11_MODULE(backend, m) {
    infini::register_operator_timer(m);
    infini::export_values(m);
    infini::export_functions(m);
    infini::init_graph_builder(m);
}
