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

} // namespace infini