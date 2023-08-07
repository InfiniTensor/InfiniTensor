#include "core/graph_builder.h"

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
    std::vector<Tensor> inputTensors;
    inputTensors.reserve(inputs.size());
    for (auto const &name : inputs) {
        if (tensors.find(name) != tensors.end()) {
            inputTensors.push_back(tensors[name]);
        } else {
            unknownInputs.insert(name);
        }
    }
    if (unknownInputs.empty()) {
        // TODO
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
