#include "core/graph.h"
#include "core/kernel.h"

namespace infini {
namespace {
bool isShapeOperation(OpType type) {
    switch (type.underlying()) {
    case OpType::Shape:
    case OpType::Gather:
    case OpType::Unsqueeze:
    case OpType::Squeeze:
    case OpType::Concat:
    case OpType::Cast:
    case OpType::Identity:
    case OpType::Reshape:
        return true;
    default:
        return false;
    }
}
} // namespace

void GraphObj::evaluateShapeOperator(const Operator &op) {
    bool derivedFromShape = op->getOpType() == OpType::Shape;
    bool inputsAvailable = true;
    for (const auto &input : op->getInputs()) {
        derivedFromShape |= isShapeTensor(input);
        inputsAvailable &=
            shapeTensorValues.count(input.get()) || input->isWeight();
    }
    if (op->getOpType() == OpType::Shape ||
        (derivedFromShape && inputsAvailable &&
         isShapeOperation(op->getOpType())))
        evaluateShapeTensor(op->getOutput());
}

Tensor GraphObj::evaluateShapeTensor(const Tensor &tensor) {
    auto found = shapeTensorValues.find(tensor.get());
    if (found != shapeTensorValues.end())
        return found->second;
    auto cpu = NativeCpuRuntimeObj::getInstance();
    auto source = tensor->getSource();
    if (!source) {
        IT_ASSERT(tensor->hasData(),
                  "Shape tensor source needs a value before shape inference");
        auto host = tensor->clone(cpu);
        shapeTensorValues.emplace(tensor.get(), host);
        return host;
    }
    IT_ASSERT(isShapeOperation(source->getOpType()),
              "Unsupported data-dependent shape operator: " +
                  source->toString());
    TensorVec hostInputs;
    for (const auto &input : source->getInputs()) {
        if (source->getOpType() == OpType::Shape) {
            // Shape consumes metadata. In particular, do not copy the image
            // or execute its data-producing ancestors on the CPU.
            hostInputs.emplace_back(
                make_ref<TensorObj>(input->getDims(), input->getDType(), cpu));
        } else {
            hostInputs.emplace_back(evaluateShapeTensor(input));
        }
    }
    const auto shapes = source->inferShape(hostInputs);
    IT_ASSERT(shapes.has_value(), "Shape subgraph inference failed");
    const auto dtypes = source->inferDataType(hostInputs);
    TensorVec hostOutputs;
    for (size_t i = 0; i < shapes->size(); ++i) {
        auto output = make_ref<TensorObj>(shapes->at(i), dtypes[i], cpu);
        // Bound accidental data computation through this host-only path.
        IT_ASSERT(output->size() <= 4096,
                  "Shape tensor exceeds the supported 4096 element limit");
        output->dataMalloc();
        hostOutputs.emplace_back(output);
    }
    auto hostOp = source->clone(hostInputs, hostOutputs);
    KernelRegistry::getInstance()
        .getKernel({Device::CPU, source->getOpType().underlying()})
        ->compute(hostOp, cpu.get());
    for (size_t i = 0; i < hostOutputs.size(); ++i)
        shapeTensorValues.emplace(source->getOutput(i).get(), hostOutputs[i]);
    shapeOperators.emplace(source.get());
    return shapeTensorValues.at(tensor.get());
}

void GraphObj::uploadShapeTensors() {
    for (const auto &tensor : tensors) {
        if (isShapeTensor(tensor))
            tensor->copyData(shapeTensorValues.at(tensor.get()));
    }
}

} // namespace infini
