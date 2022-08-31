#include "core/graph.h"

namespace infini {

void GraphObj::updateConnection() { IT_TODO_HALT(); }

string GraphObj::toString() const {
    std::ostringstream oss;
    oss << "Graph Tensors:\n";
    for (const auto &tensor : tensors)
        oss << tensor << "\n";

    oss << "Graph operators:\n";
    for (const auto &op : ops)
        oss << op << "\n";
    return oss.str();
}

void GraphObj::dataMalloc() {
    for (auto &tensor : tensors) {
        tensor->dataMalloc();
    }
}

Tensor GraphObj::addTensor(Shape dim, DataType dtype) {
    Tensor tensor = make_ref<TensorObj>(dim, dtype, runtime);
    tensors.emplace_back(tensor);
    return tensor;
}

OpVec GraphObj::getComputeOps() const {
    OpVec opList;
    for (auto op : ops)
        if (op->isComputeOp())
            opList.emplace_back(op);
    return opList;
};

} // namespace infini