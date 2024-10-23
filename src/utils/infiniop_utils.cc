#include "utils/infiniop_utils.h"

namespace infini {

DataLayout toInfiniopDataLayout(int dataType) {
    switch (dataType) {
    case 1:
        return F32;
    case 4:
        return F16;
    default:
        IT_TODO_HALT();
    };
}

vector<uint64_t> toInfiniopShape(const vector<int> &shape) {
    vector<uint64_t> infiniShape;
    for (auto s : shape) {
        infiniShape.push_back(s);
    }
    return infiniShape;
}

} // namespace infini
