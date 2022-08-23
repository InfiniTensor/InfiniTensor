#include "core/graph.h"

namespace infini {

void GraphObj::updateConnection() { IT_TODO_HALT(); }

string GraphObj::toString() const {
    std::ostringstream oss;
    oss << "Graph operators:\n";
    for (const auto &op : ops)
        oss << op << "\n";
    return oss.str();
}

void GraphObj::dataMalloc() {
    for (auto &tensor : tensors) {
        tensor->dataMalloc(runtime);
    }
}

Tensor GraphObj::addTensor(Shape dim, DataType dtype) {
    Tensor tensor = make_ref<TensorObj>(dim, dtype, runtime);
    tensors.emplace_back(tensor);
    return tensor;
}

} // namespace infini