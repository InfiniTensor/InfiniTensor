#include "utils/operator_utils.h"
#include "core/runtime.h"

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

std::string get_kernel_attrs_str(const KernelAttrs &kernelAttrs) {
    std::string deviceStr = "UNKNOWN_DEVICE";
    switch (std::get<0>(kernelAttrs)) {
    case Device::CPU:
        deviceStr = "CPU";
        break;
    case Device::CUDA:
        deviceStr = "CUDA";
        break;
    case Device::BANG:
        deviceStr = "BANG";
        break;
    case Device::INTELCPU:
        deviceStr = "INTELCPU";
        break;
    case Device::KUNLUN:
        deviceStr = "KUNLUN";
        break;
    default:
        deviceStr = "UNKNOWN_DEVICE";
    }
    std::string opStr = OpType(std::get<1>(kernelAttrs)).toString();
    std::string datatypeStr = std::get<2>(kernelAttrs).toString();
    return deviceStr + ", " + opStr + ", " + datatypeStr;
}
} // namespace infini
