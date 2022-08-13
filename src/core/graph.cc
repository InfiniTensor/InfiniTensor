#include "core/graph.h"

namespace infini {

void GraphNode::updateConnection() { IT_TODO_HALT(); }

string GraphNode::toString() const {
    std::ostringstream oss;
    oss << "GraphNode operators:\n";
    for (const auto &op : ops)
        oss << op << "\n";
    return oss.str();
}

void GraphNode::dataMalloc() {
    for (auto &tensor : tensors)
        tensor->dataMalloc();
}

Tensor GraphNode::addTensor(Shape dim, DataType dtype) {
    Tensor tensor = make_ref<TensorNode>(dim, dtype);
    tensors.emplace_back(tensor);
    return tensor;
}

} // namespace infini