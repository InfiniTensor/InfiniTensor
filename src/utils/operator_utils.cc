#include "utils/operator_utils.h"

namespace infini {

Shape infer_broadcast(const Shape &A, const Shape &B) {
    if (A.empty() && B.empty()) {
        return {};
    }
    auto A_ = A;
    auto B_ = B;
    int rankA = A.size();
    int rankB = B.size();
    int rank = std::max(rankA, rankB);
    if (rankA < rank) {
        for (int i = 0; i < rank - rankA; ++i) {
            A_.insert(A_.begin(), 1);
        }
    }
    if (rankB < rank) {
        for (int i = 0; i < rank - rankB; ++i) {
            B_.insert(B_.begin(), 1);
        }
    }
    Shape ret;
    for (int i = 0; i < rank; ++i) {
        IT_ASSERT(A_[i] == B_[i] || A_[i] == 1 || B_[i] == 1);
        auto shapeEle = std::max(A_[i], B_[i]);
        ret.emplace_back(shapeEle);
    }
    return ret;
}

int get_real_axis(const int &axis, const int &rank) {
    IT_ASSERT(rank >= 1);
    IT_ASSERT(axis >= -rank && axis <= (rank - 1));
    int newAxis;
    if (axis < 0) {
        newAxis = rank + axis;
    } else {
        newAxis = axis;
    }
    return newAxis;
}

bool is_unidirectional_broadcasting(const Shape &A, const Shape &B) {
    // check if tensor B is unidirectional broadcastable to tensor A
    auto B_ = B;
    int rankA = A.size();
    int rankB = B.size();
    if (rankA < rankB) {
        return false;
    }
    if (rankA > rankB) {
        for (auto i = 0; i < rankA - rankB; ++i) {
            B_.insert(B_.begin(), 1);
        }
    }
    for (auto i = 0; i < rankA; ++i) {
        if (A[i] == B_[i] || B_[i] == 1) {
            continue;
        } else {
            return false;
        }
    }
    return true;
}

Shape locate_index(size_t inputN, Shape const &shape) {
    Shape ans(shape.size());
    auto i = ans.rbegin();
    auto j = shape.rbegin(), ej = shape.rend();
    while (j != ej) {
        auto div = std::div(inputN, *j++);
        *i++ = div.rem;
        inputN = div.quot;
    }
    return ans;
}

size_t delocate_index(Shape const &shapeIndex, vector<int> const &shape,
                      vector<int> const &stride) {
    size_t ans = 0;
    vector<int> index(shapeIndex.size());
    IT_ASSERT(shapeIndex.size() == shape.size());
    IT_ASSERT(shape.size() == stride.size());
    for (size_t i = 0; i < shape.size(); ++i) {
        index[i] = shapeIndex[i] % shape[i];
        ans += index[i] * stride[i];
    }
    return ans;
}
} // namespace infini
