#include "core/common.h"

namespace infini {

inline HashType hashAppend(HashType a, HashType b) {
    return (a * 10000019 + b * 10000079) % 2147483647;
}

// inline HashType hashPack(HashType x) { return (x * 10000103) % 2147483647; }

template <typename T> inline HashType hashVector(const vector<T> &vec) {
    HashType ret = 0;
    for (auto v : vec)
        ret = hashAppend(ret, v);
    return ret;
}

} // namespace infini
