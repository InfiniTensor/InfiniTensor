#include <core/tensor.h>
namespace it {

string TensorBaseNode::toString() const {
    return "TensorBaseNode " + std::to_string(guid);
}

}; // namespace it