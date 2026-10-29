#include "lowering.h"
#include "core/kernel.h"
#include "operators/batch_norm.h"
#include "operators/concat.h"
#include "operators/conv.h"
#include "operators/expand.h"
#include "operators/gather.h"
#include "operators/infini_op.h"
#include "operators/layer_norm.h"
#include "operators/matmul.h"
#include "operators/pooling.h"
#include "operators/reduce.h"
#include "operators/rope.h"
#include "operators/slice.h"
#include "operators/softmax.h"
#include "operators/transpose.h"
#include "operators/unary.h"
#include "tensor_view.h"
#include "utils/data_convert.h"
#include <array>
#include <cmath>
#include <cstring>

namespace infini::infiniops {
namespace {
TensorView view(const Tensor &tensor, const RuntimeObj *runtime) {
    return toInfiniOpsTensor(tensor, runtime);
}
TensorView temporary(Builder &builder, const RuntimeObj *runtime,
                     const Shape &shape, const ::infini::DataType &dtype) {
    size_t count = 1;
    ::infini::rt::TensorView::Shape dims;
    for (auto dim : shape) {
        IT_ASSERT(dim >= 0);
        dims.push_back(dim);
        count *= dim;
    }
    auto allocation = builder.temporary(count * dtype.getSize());
    return makeInfiniOpsTensor(allocation.get(), dims,
                               toInfiniOpsDataType(dtype), runtime);
}
TensorView scalar(Builder &builder, const RuntimeObj *runtime,
                  const ::infini::DataType &dtype, double number) {
    std::array<uint8_t, sizeof(double)> data{};
    if (dtype == ::infini::DataType::Float32) {
        const float v = static_cast<float>(number);
        std::memcpy(data.data(), &v, sizeof(v));
    } else if (dtype == ::infini::DataType::Double) {
        std::memcpy(data.data(), &number, sizeof(number));
    } else if (dtype == ::infini::DataType::Float16 ||
               dtype == ::infini::DataType::BFloat16) {
        const auto v = dtype == ::infini::DataType::Float16
                           ? float_to_fp16(number)
                           : float_to_bfp16(number);
        std::memcpy(data.data(), &v, sizeof(v));
    } else
        IT_ASSERT(false, "HardSigmoid requires floating point input");
    auto storage = builder.constant(dtype.getSize(), [&](void *ptr) {
        runtime->copyBlobFromCPU(ptr, data.data(), dtype.getSize());
    });
    return makeInfiniOpsTensor(storage.get(), {}, toInfiniOpsDataType(dtype),
                               runtime);
}
TensorView transposed(TensorView tensor, bool transpose) {
    if (!transpose)
        return tensor;
    TensorView::Shape shape(tensor.shape().begin(), tensor.shape().end());
    TensorView::Strides strides(tensor.strides().begin(),
                                tensor.strides().end());
    IT_ASSERT(shape.size() >= 2, "Matmul transpose requires rank >= 2");
    std::swap(shape[shape.size() - 1], shape[shape.size() - 2]);
    std::swap(strides[strides.size() - 1], strides[strides.size() - 2]);
    return {tensor.data(), shape, tensor.dtype(), tensor.device(), strides};
}
Value jsonValue(const nlohmann::json &value) {
    if (value.is_null())
        return std::monostate{};
    if (value.is_boolean())
        return value.get<bool>();
    if (value.is_number_unsigned()) {
        IT_ASSERT(value.get<uint64_t>() <=
                      uint64_t(std::numeric_limits<int64_t>::max()),
                  "Integer attribute out of range");
        return value.get<int64_t>();
    }
    if (value.is_number_integer())
        return value.get<int64_t>();
    if (value.is_number_float())
        return value.get<double>();
    if (value.is_string())
        return value.get<std::string>();
    if (value.is_array()) {
        std::vector<int64_t> result;
        for (const auto &item : value) {
            IT_ASSERT(item.is_number_integer(),
                      "Only integer vector attributes are supported");
            result.push_back(std::get<int64_t>(jsonValue(item)));
        }
        return result;
    }
    IT_ASSERT(false, "Unsupported InfiniOp attribute type");
    return {};
}
void generic(const InfiniOpObj &op, const RuntimeObj *runtime,
             Builder &builder) {
    NamedValues values;
    for (const auto &[name, range] : op.getInputBindings()) {
        std::vector<TensorView> tensors;
        for (size_t i = 0; i < range.second; ++i)
            tensors.push_back(view(op.getInputs(range.first + i), runtime));
        values.emplace(name,
                       range.second == 1 ? Value{tensors[0]} : Value{tensors});
    }
    for (const auto &[name, range] : op.getOutputBindings()) {
        std::vector<TensorView> tensors;
        for (size_t i = 0; i < range.second; ++i)
            tensors.push_back(view(op.getOutput(range.first + i), runtime));
        values.emplace(name,
                       range.second == 1 ? Value{tensors[0]} : Value{tensors});
    }
    for (const auto &item : op.getAttributes().items())
        values.emplace(item.key(), jsonValue(item.value()));
    builder.emit(op.getOperation(), values);
}
size_t readMaxPosition(const Tensor &pos, const RuntimeObj *context) {
    IT_ASSERT(pos != nullptr);
    size_t maxPos = 0;
    const auto count = pos->size();

    if (pos->getDType() == ::infini::DataType::Int32) {
        std::vector<int32_t> host(count);
        context->copyBlobToCPU(host.data(), pos->getRawDataPtr<void *>(),
                               pos->getBytes());
        for (auto value : host) {
            IT_ASSERT(value >= 0);
            maxPos = std::max(maxPos, static_cast<size_t>(value));
        }
        return maxPos;
    }

    if (pos->getDType() == ::infini::DataType::Int64) {
        std::vector<int64_t> host(count);
        context->copyBlobToCPU(host.data(), pos->getRawDataPtr<void *>(),
                               pos->getBytes());
        for (auto value : host) {
            IT_ASSERT(value >= 0);
            maxPos = std::max(maxPos, static_cast<size_t>(value));
        }
        return maxPos;
    }

    IT_TODO_HALT_MSG("RoPE position ids must be int32 or int64");
}

void buildRopeTables(size_t tableLen, size_t tableDim, size_t headDim,
                     std::vector<float> &sinTable,
                     std::vector<float> &cosTable) {
    sinTable.resize(tableLen * tableDim);
    cosTable.resize(tableLen * tableDim);

    for (size_t pos = 0; pos < tableLen; ++pos) {
        for (size_t i = 0; i < tableDim; ++i) {
            const auto invFreq =
                std::pow(10000.0f, -static_cast<float>(2 * i) /
                                       static_cast<float>(headDim));
            const auto angle = static_cast<float>(pos) * invFreq;
            sinTable[pos * tableDim + i] = std::sin(angle);
            cosTable[pos * tableDim + i] = std::cos(angle);
        }
    }
}

void lowerRope(const Operator &node, const RuntimeObj *context,
               Builder &builder) {
    auto op = as<RoPEObj>(node);
    auto pos = op->getInputs(0);
    auto input = op->getInputs(1);
    auto output = op->getOutput();

    const auto inputDims = input->getDims();
    const auto posDims = pos->getDims();
    const auto outputDims = output->getDims();
    IT_ASSERT(inputDims.size() == 3);
    IT_ASSERT(posDims.size() == 2);
    IT_ASSERT(outputDims == inputDims);
    IT_ASSERT(inputDims[0] == posDims[0]);
    IT_ASSERT(inputDims[1] == posDims[1]);

    const size_t batch = static_cast<size_t>(inputDims[0]);
    const size_t seqLen = static_cast<size_t>(inputDims[1]);
    const size_t hidden = static_cast<size_t>(inputDims[2]);
    const size_t headDim = 128;
    IT_ASSERT(hidden % headDim == 0);
    const size_t numHeads = hidden / headDim;
    const size_t tableDim = headDim / 2;

    auto maxPos = readMaxPosition(pos, context);
    auto tableLen = std::max(seqLen, maxPos + 1);
    std::vector<float> sinHost;
    std::vector<float> cosHost;
    buildRopeTables(tableLen, tableDim, headDim, sinHost, cosHost);

    const auto tableBytes = tableLen * tableDim * sizeof(float);
    auto sinBlob = builder.constant(tableBytes, [&](void *ptr) {
        context->copyBlobFromCPU(ptr, sinHost.data(), tableBytes);
    });
    auto cosBlob = builder.constant(tableBytes, [&](void *ptr) {
        context->copyBlobFromCPU(ptr, cosHost.data(), tableBytes);
    });

    ::infini::rt::TensorView::Shape ropeShape{batch, seqLen, numHeads, headDim};
    ::infini::rt::TensorView::Strides ropeStrides{
        static_cast<::infini::rt::TensorView::Stride>(seqLen * hidden),
        static_cast<::infini::rt::TensorView::Stride>(hidden),
        static_cast<::infini::rt::TensorView::Stride>(headDim), 1};
    ::infini::rt::TensorView::Shape tableShape{tableLen, tableDim};

    auto inputView = infiniops::makeInfiniOpsTensor(
        input->getRawDataPtr<void *>(), ropeShape,
        infiniops::toInfiniOpsDataType(input->getDType()), context,
        ropeStrides);
    auto posView = infiniops::toInfiniOpsTensor(pos, context);
    auto sinView = infiniops::makeInfiniOpsTensor(
        sinBlob.get(), tableShape, ::infini::rt::DataType::kFloat32, context);
    auto cosView = infiniops::makeInfiniOpsTensor(
        cosBlob.get(), tableShape, ::infini::rt::DataType::kFloat32, context);
    auto outputView = infiniops::makeInfiniOpsTensor(
        output->getRawDataPtr<void *>(), ropeShape,
        infiniops::toInfiniOpsDataType(output->getDType()), context,
        ropeStrides);

    builder.emit("RotaryEmbeddingInfinilm", inputView, posView, sinView,
                 cosView, false, outputView);
}

bool lower(const Operator &node, const RuntimeObj *runtime, Builder &builder) {
    if (const auto op = dynamic_cast<const InfiniOpObj *>(node.get())) {
        generic(*op, runtime, builder);
        return true;
    }
    const auto type = node->getOpType();
    const auto input = [&](size_t i = 0) {
        return view(node->getInputs(i), runtime);
    };
    const auto output = [&] { return view(node->getOutput(), runtime); };
    if (type == OpType::Reshape || type == OpType::Flatten ||
        type == OpType::Identity || type == OpType::Squeeze ||
        type == OpType::Unsqueeze) {
        auto *src = node->getInputs(0).get();
        auto *dst = node->getOutput().get();
        builder.plan.steps.push_back({[runtime, src, dst](const Handle &) {
                                          runtime->copyBlob(dst, src);
                                      },
                                      true, "Copy", "InfiniRT"});
        return true;
    }
    if (type == OpType::MatMul) {
        const auto op = as<MatmulObj>(node);
        IT_ASSERT(op->getAct() == ActType::None,
                  "Fused Matmul activation is unsupported");
        builder.emit("Matmul", transposed(input(0), op->getTransA()),
                     transposed(input(1), op->getTransB()), output());
        if (op->getBias())
            builder.emit("Add", output(),
                         toInfiniOpsBroadcastTensor(op->getBias(),
                                                    op->getOutput()->getDims(),
                                                    runtime),
                         1.0, output());
        return true;
    }
    if (type == OpType::Conv || type == OpType::Conv3d) {
        std::vector<int64_t> stride, pad, dilation;
        int groups;
        ActType activation;
        if (type == OpType::Conv) {
            const auto op = as<ConvObj>(node);
            const auto [ph, pw, sh, sw, dh, dw] = op->getPadStrideDilation();
            stride = {sh, sw};
            pad = {ph, pw};
            dilation = {dh, dw};
            groups = op->getNumGroups();
            activation = op->getAct();
        } else {
            const auto op = as<Conv3dObj>(node);
            const auto [pd, ph, pw, sd, sh, sw, dd, dh, dw] =
                op->getPadStrideDilation();
            stride = {sd, sh, sw};
            pad = {pd, ph, pw};
            dilation = {dd, dh, dw};
            groups = op->getNumGroups();
            activation = op->getAct();
        }
        builder.emit("Convolution", input(0), input(1),
                     std::optional<TensorView>{}, stride, pad, dilation, false,
                     std::vector<int64_t>(stride.size(), 0), int64_t(groups),
                     output());
        if (activation == ActType::Relu)
            builder.emit("Relu", output(), output());
        else if (activation == ActType::Sigmoid)
            builder.emit("Sigmoid", output(), output());
        else
            IT_ASSERT(activation == ActType::None,
                      "Unsupported Conv activation");
        return true;
    }
    if (type == OpType::RoPE) {
        // Position IDs may be produced by an earlier node. Resolve them at
        // this step, after its producers have executed, never during prepare.
        WRef<OperatorObj> weak = node;
        builder.plan.steps.push_back(
            {[weak, runtime](const Handle &handle) {
                 auto live = weak.lock();
                 IT_ASSERT(live != nullptr, "Prepared graph was destroyed");
                 const auto *infini =
                     dynamic_cast<const InfiniRuntimeObj *>(runtime);
                 IT_ASSERT(infini != nullptr);
                 infini->sync();
                 Builder late([infini](size_t bytes) {
                     return infini->acquireWorkspace(bytes);
                 });
                 try {
                     lowerRope(live, runtime, late);
                     late.finalize();
                     late.plan.run(handle);
                     infini->sync();
                 } catch (...) {
                     try {
                         infini->sync();
                     } catch (...) {
                     }
                     throw;
                 }
             },
             false, "RoPE", "runtime position metadata"});
        return true;
    }
    if (type == OpType::Gather) {
        const auto op = as<GatherObj>(node);
        builder.emit("OnnxGather", input(0), input(1), int64_t(op->getAxis()),
                     output());
        return true;
    }
    if (type == OpType::LayerNormalization) {
        const auto op = as<LayerNormObj>(node);
        IT_ASSERT(op->getDType() == ::infini::DataType::Float32 &&
                      op->getStashType() == 1,
                  "ONNX LayerNormalization currently requires FP32 input and "
                  "stash_type=1");
        const auto dims = op->getInputs(0)->getDims();
        const auto axis = op->getAxis();
        const Shape normalized(dims.begin() + axis, dims.end());
        Shape stats = dims;
        std::fill(stats.begin() + axis, stats.end(), 1);
        std::optional<TensorView> bias;
        if (op->getBias())
            bias.emplace(
                toInfiniOpsBroadcastTensor(op->getBias(), normalized, runtime));
        builder.emit(
            "NativeLayerNorm", input(0),
            std::vector<int64_t>(normalized.begin(), normalized.end()),
            std::optional<TensorView>{toInfiniOpsBroadcastTensor(
                op->getInputs(1), normalized, runtime)},
            bias, double(op->getEps()), output(),
            temporary(builder, runtime, stats, ::infini::DataType::Float32),
            temporary(builder, runtime, stats, ::infini::DataType::Float32));
        return true;
    }
    if (type == OpType::RMSNorm) {
        builder.emit("RmsNorm", input(0), input(1), 1e-6f, output());
        return true;
    }
    if (type == OpType::Softmax) {
        const auto op = as<SoftmaxObj>(node);
        builder.emit("Softmax", input(), int64_t(op->getAxis()),
                     std::optional<DataType>{}, output());
        return true;
    }
    if (type == OpType::Concat) {
        const auto op = as<ConcatObj>(node);
        std::vector<TensorView> tensors;
        for (const auto &t : op->getInputs())
            tensors.push_back(view(t, runtime));
        builder.emit("Cat", tensors, int64_t(op->getDim()), output());
        return true;
    }
    if (type == OpType::ReduceMean) {
        const auto op = as<ReduceMeanObj>(node);
        const auto dims = op->getAxes();
        std::vector<int64_t> axes(dims.begin(), dims.end());
        builder.emit("Mean", input(), std::optional<std::vector<int64_t>>{axes},
                     op->getKeepDims(), std::optional<DataType>{}, output());
        return true;
    }
    if (type == OpType::BatchNormalization) {
        const auto op = as<BatchNormObj>(node);
        IT_ASSERT(!op->getTrainingMode(), "Training BatchNorm is unsupported");
        const Shape saved{static_cast<int>(op->getInputs(1)->size())};
        builder.emit(
            "NativeBatchNorm", input(0), std::optional<TensorView>{input(3)},
            std::optional<TensorView>{input(4)},
            std::optional<TensorView>{input(1)},
            std::optional<TensorView>{input(2)}, false,
            double(op->getMomentum()), double(op->getEps()), output(),
            temporary(builder, runtime, saved, ::infini::DataType::Float32),
            temporary(builder, runtime, saved, ::infini::DataType::Float32));
        return true;
    }
    if (type == OpType::MaxPool || type == OpType::AveragePool) {
        const auto op = as<PoolingObj>(node);
        const auto [ph, pw, sh, sw, dh, dw] = op->getPadStrideDilation();
        const std::vector<int64_t> kernel{op->getKh(), op->getKw()},
            stride{sh, sw}, padding{ph, pw};
        if (type == OpType::MaxPool)
            builder.emit("MaxPool2dWithIndices", input(), kernel, stride,
                         padding, std::vector<int64_t>{dh, dw},
                         op->getCeilMode() != 0, output(),
                         temporary(builder, runtime, op->getOutput()->getDims(),
                                   ::infini::DataType::Int64));
        else {
            IT_ASSERT(dh == 1 && dw == 1, "AvgPool dilation is unsupported");
            builder.emit("AvgPool2d", input(), kernel, stride, padding,
                         op->getCeilMode() != 0, true, std::optional<int64_t>{},
                         output());
        }
        return true;
    }
    if (type == OpType::Transpose) {
        const auto op = as<TransposeObj>(node);
        const auto perm = op->getPermute();
        builder.emit("PermuteCopy", input(),
                     std::vector<int64_t>(perm.begin(), perm.end()), output());
        return true;
    }
    if (type == OpType::Expand) {
        const auto op = as<ExpandObj>(node);
        const auto shape = op->getShape();
        builder.emit("ExpandCopy", input(),
                     std::vector<int64_t>(shape.begin(), shape.end()), false,
                     output());
        return true;
    }
    if (type == OpType::Slice) {
        const auto op = as<SliceObj>(node);
        const auto starts = op->getStarts(), steps = op->getSteps();
        const auto dims = op->getOutput()->getDims(),
                   strides = op->getInputs(0)->getStride();
        IT_ASSERT(starts.size() == dims.size() && steps.size() == dims.size());
        size_t offset = 0;
        TensorView::Shape shape;
        TensorView::Strides viewStrides;
        for (size_t i = 0; i < dims.size(); ++i) {
            IT_ASSERT(starts[i] >= 0 && steps[i] > 0,
                      "Negative Slice steps are unsupported");
            offset += starts[i] * strides[i];
            shape.push_back(dims[i]);
            viewStrides.push_back(strides[i] * steps[i]);
        }
        auto ptr = op->getInputs(0)->getRawDataPtr<uint8_t *>() +
                   offset * op->getDType().getSize();
        auto sliced =
            makeInfiniOpsTensor(ptr, shape, toInfiniOpsDataType(op->getDType()),
                                runtime, viewStrides);
        builder.emit("SliceCopy", sliced, int64_t{0}, std::optional<int64_t>{},
                     std::optional<int64_t>{}, int64_t{1}, output());
        return true;
    }
    if (type == OpType::Clip) {
        const auto op = as<ClipObj>(node);
        builder.emit(
            "Clip", input(),
            op->getMin() ? std::optional<double>{*op->getMin()} : std::nullopt,
            op->getMax() ? std::optional<double>{*op->getMax()} : std::nullopt,
            output());
        return true;
    }
    if (type == OpType::HardSigmoid) {
        const auto op = as<HardSigmoidObj>(node);
        const auto temp = temporary(builder, runtime,
                                    op->getOutput()->getDims(), op->getDType());
        builder.emit("Mul", input(),
                     scalar(builder, runtime, op->getDType(), op->getAlpha()),
                     temp);
        builder.emit("Add", temp,
                     scalar(builder, runtime, op->getDType(), op->getBeta()),
                     1.0, temp);
        builder.emit("Clip", temp, std::optional<double>{0},
                     std::optional<double>{1}, output());
        return true;
    }
    if (type == OpType::Gelu) {
        builder.emit("Gelu", input(), std::string("none"), output());
        return true;
    }
    if (type == OpType::Sub || type == OpType::Div) {
        const auto dims = node->getOutput()->getDims();
        const auto lhs =
            toInfiniOpsBroadcastTensor(node->getInputs(0), dims, runtime);
        const auto rhs =
            toInfiniOpsBroadcastTensor(node->getInputs(1), dims, runtime);
        if (type == OpType::Sub)
            builder.emit("Sub", lhs, rhs, 1.0, output());
        else if (node->getDType() == ::infini::DataType::Int32 ||
                 node->getDType() == ::infini::DataType::Int64)
            builder.emit("Div", lhs, rhs, std::optional<std::string>{"trunc"},
                         output());
        else
            builder.emit("Div", lhs, rhs, output());
        return true;
    }
    if (type == OpType::Add || type == OpType::Mul) {
        const auto dims = node->getOutput()->getDims();
        builder.emit(
            type == OpType::Add ? "Add" : "Mul",
            toInfiniOpsBroadcastTensor(node->getInputs(0), dims, runtime),
            toInfiniOpsBroadcastTensor(node->getInputs(1), dims, runtime),
            output());
        return true;
    }
    // Same-name, attribute-free tensor signatures need no handwritten adapter.
    const auto name = std::string(type.toString());
    if ((type.isUnary() || type.isBinary()) &&
        Registry::instance().simple(name, node->getInputs().size(),
                                    node->getOutputs().size())) {
        Values values;
        for (const auto &t : node->getInputs())
            values.push_back(view(t, runtime));
        values.push_back(output());
        builder.emit(name, values);
        return true;
    }
    return false;
}
} // namespace
std::shared_ptr<PreparedGraph> prepareGraph(const Graph &graph,
                                            const RuntimeObj *runtime,
                                            MemoryAllocator allocate) {
    if (!allocate)
        allocate = [runtime](size_t bytes) {
            const auto *infini = dynamic_cast<const InfiniRuntimeObj *>(runtime);
            IT_ASSERT(infini != nullptr,
                      "Prepared execution requires an Infini runtime");
            return infini->acquireWorkspace(bytes);
        };
    const auto lowerAll = [&](Builder &builder) {
        for (const auto &op : graph->getOperators()) {
            builder.beginNode();
            const auto first = builder.plan.steps.size();
            try {
                if (!lower(op, runtime, builder)) {
                    const auto attrs = KernelAttrs{ExecutionProvider::Infini,
                                                   op->getOpType().underlying()};
                    auto *kernel = KernelRegistry::getInstance().getKernel(attrs);
                    WRef<OperatorObj> weak = op;
                    builder.plan.steps.push_back(
                        {[kernel, weak, runtime](const Handle &) {
                             auto live = weak.lock();
                             IT_ASSERT(live != nullptr,
                                       "Prepared graph was destroyed");
                             kernel->compute(live, runtime);
                         },
                         kernel->captureSafe(), op->getOpType().toString(),
                         typeid(*kernel).name()});
                }
                for (size_t i = first; i < builder.plan.steps.size(); ++i)
                    builder.plan.steps[i].nodeId = op->getGuid();
            } catch (const std::exception &error) {
                throw Exception("InfiniOps prepare failed at " + op->toString() +
                                ": " + error.what());
            }
        }
    };
    // Fix helper addresses before descriptors borrow them; the layout pass
    // does not construct providers, upload constants or allocate device memory.
    size_t temporaryBytes;
    {
        Builder layout(allocate, 0, true);
        lowerAll(layout);
        temporaryBytes = layout.temporaryPeak();
    }
    Builder builder(std::move(allocate), temporaryBytes);
    lowerAll(builder);
    builder.finalize();
    auto prepared = std::make_shared<PreparedGraph>();
    prepared->owner = graph;
    prepared->graphId = graph->getCaptureStateId();
    prepared->generation = graph->getCaptureGeneration();
    prepared->plan = std::move(builder.plan);
    return prepared;
}
} // namespace infini::infiniops
