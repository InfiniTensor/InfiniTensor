#include "utils/infiniop_utils.h"

namespace infini {

DataLayout toInfiniopDataLayout(int dataType) {
    switch (dataType) {
    case 1:
        return F32;
    case 2:
        return U8;
    case 3:
        return I8;
    case 4:
        return U16;
    case 5:
        return I16;
    case 6:
        return I32;
    case 7:
        return I64;
    case 10:
        return F16;
    case 11:
        return F64;
    case 12:
        return U32;
    case 13:
        return U64;
    case 14:
        return BF16;
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
