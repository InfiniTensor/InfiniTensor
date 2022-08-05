#include "core/graph.h"

namespace it {

void GraphNode::updateConnection() { IT_TODO_HALT(); }

string GraphNode::toString() const {
    std::ostringstream oss;
    oss << "GraphNode operators:\n";
    for (const auto &op : ops)
        oss << op << "\n";
    return oss.str();
}

} // namespace it