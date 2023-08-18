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

using namespace refactor;
graph::NodeInfo getNodeInfo(const Operator &obj) {
	auto type = obj->getOpType().underlying();
	graph::NodeInfo nodeInfo{common::OpType::Unknown};
#define CASE(T)																				\
		case OpType::T:																		\
			nodeInfo = {common::OpType::T, refactor::graph::Attributes{}};		\
			break;

	switch (type) {
			case OpType::MatMul: {
					auto matmul = dynamic_cast<const MatmulObj *>(obj.get());
					auto transA = matmul->getTransA();
					auto transB = matmul->getTransB();
					nodeInfo = {common::OpType::MatMul,
							{{"transA", static_cast<int>(transA)},
									{"transB", static_cast<int>(transB)}}};
					break;
			}
			CASE(Relu)
			default :
				IT_TODO_HALT_MSG("Don't Support OpType");
	}
#undef CASE
	return nodeInfo;
}
} // namespace infini
