#include "core/graph_builder.h"
#include "operators/element_wise.h"
#include "operators/pad.h"
#include "operators/pooling.h"
#include "operators/unary.h"

namespace infini {

void GraphBuilder::addTensor( //
    std::string name,         //
    Shape shape,              //
    DataType dataType         //
) {
    if (tensors.find(name) != tensors.end()) {
        graph->addTensor(tensors[name] = std::make_shared<TensorObj>(
                             shape, dataType, graph->getRuntime()));
    }
}

void GraphBuilder::addOperation(      //
    std::string opType,               //
    std::vector<std::string> inputs,  //
    std::vector<std::string> outputs, //
    Attributes attributes             //
) {
    std::unordered_set<std::string> unknownInputs;
    std::vector<Tensor> inputTensors, outputTensors;
    inputTensors.reserve(inputs.size());
    for (auto const &name : inputs) {
        if (tensors.find(name) != tensors.end()) {
            inputTensors.push_back(tensors[name]);
        } else {
            unknownInputs.insert(name);
        }
    }
    if (unknownInputs.empty()) {
        if (opType == "Abs") {
            outputTensors =
                graph->addOp<AbsObj>(inputTensors[0], nullptr)->getOutputs();
        } else if (opType == "Add") {
            outputTensors =
                graph->addOp<AddObj>(inputTensors[0], inputTensors[1], nullptr)
                    ->getOutputs();
        } else if (opType == "AveragePool") {
            int kh, kw, dh = 1, dw = 1, ph, pw, sh = 0, sw = 0;
            { // `kernel_shpae`: required
                auto k = takeAttribute<Ints>(attributes, "kernel_shape");
                kh = k->at(0);
                kw = k->at(1);
            }
            // `pads`
            if (auto p = takeAttribute<Ints>(attributes, "pads"); p) {
                if (p->at(0) != p->at(2) || p->at(1) != p->at(3)) {
                    std::vector<int> padVal(p->begin(), p->end());
                    auto pad =
                        graph->addOp<PadObj>(inputTensors[0], nullptr, padVal,
                                             std::vector<int>{-2, -1});
                    inputTensors[0] = pad->getOutputs()[0];
                    ph = pw = 0;
                } else {
                    ph = p->at(0);
                    pw = p->at(1);
                }
            } else {
                ph = pw = 0;
            }
            // `strides`
            if (auto s = takeAttribute<Ints>(attributes, "strides"); s) {
                sh = s->at(0);
                sw = s->at(1);
            } else {
                sh = sw = 1;
            }
            outputTensors =
                graph
                    ->addOp<AvgPoolObj>(inputTensors[0], nullptr, kh, kw, dh,
                                        dw, ph, pw, sh, sw)
                    ->getOutputs();
        } else if (opType == "BatchNormalization") {
        } else if (opType == "Cast") {
        } else if (opType == "Clip") {
        } else if (opType == "Concat") {
        } else if (opType == "Conv") {
        } else if (opType == "ConvTranspose") {
        } else if (opType == "Div") {
            outputTensors =
                graph->addOp<DivObj>(inputTensors[0], inputTensors[1], nullptr)
                    ->getOutputs();
        } else if (opType == "Exp") {
            outputTensors =
                graph->addOp<ExpObj>(inputTensors[0], nullptr)->getOutputs();
        } else if (opType == "Expand") {
        } else if (opType == "Flatten") {
        } else if (opType == "Gather") {
        } else if (opType == "Gemm") {
        } else if (opType == "GlobalAveragePool") {
        } else if (opType == "GlobalMaxPool") {
        } else if (opType == "Greater") {
            outputTensors = graph
                                ->addOp<GreaterThanObj>(
                                    inputTensors[0], inputTensors[1], nullptr)
                                ->getOutputs();
        } else if (opType == "GreaterOrEqual") {
            outputTensors = graph
                                ->addOp<GreaterEqualObj>(
                                    inputTensors[0], inputTensors[1], nullptr)
                                ->getOutputs();
        } else if (opType == "Identity") {
        } else if (opType == "Less") {
            outputTensors = graph
                                ->addOp<LessThanObj>(inputTensors[0],
                                                     inputTensors[1], nullptr)
                                ->getOutputs();
        } else if (opType == "LessOrEqual") {
            outputTensors = graph
                                ->addOp<LessEqualObj>(inputTensors[0],
                                                      inputTensors[1], nullptr)
                                ->getOutputs();
        } else if (opType == "Log") {
        } else if (opType == "MatMul") {
        } else if (opType == "MaxPool") {
        } else if (opType == "Mul") {
            outputTensors =
                graph->addOp<MulObj>(inputTensors[0], inputTensors[1], nullptr)
                    ->getOutputs();
        } else if (opType == "Or") {
            outputTensors =
                graph->addOp<OrObj>(inputTensors[0], inputTensors[1], nullptr)
                    ->getOutputs();
        } else if (opType == "Pad") {
        } else if (opType == "Pow") {
            outputTensors =
                graph->addOp<PowObj>(inputTensors[0], inputTensors[1], nullptr)
                    ->getOutputs();
        } else if (opType == "ReduceMean") {
        } else if (opType == "Relu") {
            outputTensors =
                graph->addOp<ReluObj>(inputTensors[0], nullptr)->getOutputs();
        } else if (opType == "Reshape") {
        } else if (opType == "Sigmoid") {
            outputTensors = graph->addOp<SigmoidObj>(inputTensors[0], nullptr)
                                ->getOutputs();
        } else if (opType == "Slice") {
        } else if (opType == "Softmax") {
        } else if (opType == "Split") {
        } else if (opType == "Sqrt") {
            outputTensors =
                graph->addOp<SqrtObj>(inputTensors[0], nullptr)->getOutputs();
        } else if (opType == "Squeeze") {
        } else if (opType == "Sub") {
            outputTensors =
                graph->addOp<SubObj>(inputTensors[0], inputTensors[1], nullptr)
                    ->getOutputs();
        } else if (opType == "Tanh") {
            outputTensors =
                graph->addOp<TanhObj>(inputTensors[0], nullptr)->getOutputs();
        } else if (opType == "Transpose") {
        } else if (opType == "Unsqueeze") {
        } else if (opType == "Squeeze") {
        }
        IT_ASSERT(outputTensors.size() == outputs.size());
        for (size_t i = 0; i < outputs.size(); ++i) {
            tensors[outputs[i]] = outputTensors[i];
        }
    } else {
        unknownOps.push_back({
            std::move(opType),        //
            std::move(inputs),        //
            std::move(outputs),       //
            std::move(attributes),    //
            std::move(unknownInputs), //
        });
    }
}

} // namespace infini
