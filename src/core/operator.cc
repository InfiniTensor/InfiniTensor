#include "core/operator.h"

namespace it {

string OperatorNode::toString() const {
    std::ostringstream oss;
    oss << "Operator: ";
    return oss.str();
}

} // namespace it