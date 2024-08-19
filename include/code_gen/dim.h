#ifndef DIM_H
#define DIM_H

#include <cassert>
#include <string>
#include <vector>

namespace tpm {

using Dim = std::vector<int>;

inline Dim elementwiseMin(const Dim &lhs, const Dim &rhs) {
    assert(lhs.size() == rhs.size());
    Dim ret(lhs.size());
    for (size_t i = 0, iEnd = lhs.size(); i < iEnd; i++) {
        ret[i] = std::min(lhs[i], rhs[i]);
    }
    return ret;
}

inline Dim elementwiseMax(const Dim &lhs, const Dim &rhs) {
    assert(lhs.size() == rhs.size());
    Dim ret(lhs.size());
    for (size_t i = 0, iEnd = lhs.size(); i < iEnd; i++) {
        ret[i] = std::max(lhs[i], rhs[i]);
    }
    return ret;
}

inline Dim elementwiseAdd(const Dim &lhs, const Dim &rhs) {
    assert(lhs.size() == rhs.size());
    Dim ret(lhs.size());
    for (size_t i = 0, iEnd = lhs.size(); i < iEnd; i++) {
        ret[i] = lhs[i] + rhs[i];
    }
    return ret;
}

inline Dim elementwiseSub(const Dim &lhs, const Dim &rhs) {
    assert(lhs.size() == rhs.size());
    Dim ret(lhs.size());
    for (size_t i = 0, iEnd = lhs.size(); i < iEnd; i++) {
        ret[i] = lhs[i] - rhs[i];
    }
    return ret;
}

inline std::string dimToString(const Dim &dim) {
    std::string ret;
    ret.append("[");
    for (auto d : dim) {
        ret.append(std::to_string(d));
        ret.append(",");
    }
    ret.pop_back();
    ret.append("]");
    return ret;
}

inline Dim cntToIdx(const Dim &dim, size_t i) {
    Dim it = Dim(dim.size(), 0);
    auto tmp = i;
    int j = it.size() - 1;
    while (tmp > 0) {
        it[j] = tmp % dim[j];
        tmp /= dim[j];
        j--;
    }
    return it;
}

class DimRange {
    enum State {
        AllPos,
        Empty,
        SinglePos,
        RangePos,
        Invalid,
    };

    Dim begin, end; // closed interval [begin, end]
    State state;

    DimRange(State state) : begin({}), end({}), state(state) {}

  public:
    DimRange() : state(Invalid) {}

    DimRange(const Dim &point) : begin(point), end(point), state(SinglePos) {
        if (point.empty()) {
            state = Invalid;
        }
    }

    DimRange(const Dim &begin, const Dim &end) : begin(begin), end(end) {
        if (begin.empty() || end.empty() || begin.size() != end.size()) {
            state = Invalid;
            return;
        }
        state = SinglePos;
        for (size_t i = 0, iEnd = begin.size(); i < iEnd; ++i) {
            if (begin[i] > end[i]) {
                state = Invalid;
                return;
            }
            if (begin[i] != end[i]) {
                state = RangePos;
                return;
            }
        }
    }

    DimRange(const DimRange &rhs)
        : begin(rhs.begin), end(rhs.end), state(rhs.state){};

    Dim &getBegin() { return begin; }
    const Dim &getBegin() const { return begin; }

    Dim &getEnd() { return end; }
    const Dim &getEnd() const { return end; }

    bool isSinglePos() const { return state == SinglePos; }

    bool isEmpty() const { return state == Empty; }

    bool isAllPos() const { return state == AllPos; }

    bool notValid() const { return state == Invalid; }

    bool valid() const { return state != Invalid; }

    static DimRange getInvalid() { return DimRange(Invalid); }

    static DimRange getEmpty() { return DimRange(Empty); }

    static DimRange getAllPos() { return DimRange(AllPos); }
};

inline DimRange unionRange(const DimRange &lhs, const DimRange &rhs) {
    if (lhs.notValid() || rhs.notValid()) {
        return DimRange::getInvalid();
    }
    if (lhs.isAllPos() || rhs.isAllPos()) {
        return DimRange::getAllPos();
    }
    if (lhs.isEmpty()) {
        return DimRange(rhs);
    }
    if (rhs.isEmpty()) {
        return DimRange(lhs);
    }
    auto newBegin = elementwiseMin(lhs.getBegin(), rhs.getBegin());
    auto newEnd = elementwiseMax(lhs.getEnd(), rhs.getEnd());
    return DimRange(newBegin, newEnd);
}

} // namespace tpm

#endif // DIM_H
