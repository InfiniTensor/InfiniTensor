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
// add batchnormalization conv gemm globalaveragepool maxpool relu reshape
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
							{{"transA", static_cast<graph::Int>(transA)},
									{"transB", static_cast<graph::Int>(transB)}}};
					break;
			}
			case OpType::BatchNormalization: {
					auto batchNorm = dynamic_cast<const BatchNormObj *>(obj.get());
					auto momentum = batchNorm->getMomentum();
					auto eps = batchNorm->getEps();
					auto trainingMode = batchNorm->getTrainingMode();
					nodeInfo = {common::OpType::BatchNorm,
						{{"epsilon", static_cast<graph::Float>(eps)},
								{"momentum", static_cast<graph::Float>(momentum)},
						{"training_mode", static_cast<graph::Int>(trainingMode)}}};
					break;
			}
			case OpType::Conv: {
					auto conv = dynamic_cast<const ConvObj *>(obj.get());
					auto tuple = conv->getPadStrideDilation();
					auto group = conv->getNumGroups();
					graph::Ints pads, strides, dilations;
					{
						pads.emplace_back(std::get<0>(tuple));
						pads.emplace_back(std::get<1>(tuple));
						pads.emplace_back(std::get<0>(tuple));
						pads.emplace_back(std::get<1>(tuple));
					}
					{
						strides.emplace_back(std::get<2>(tuple));
						strides.emplace_back(std::get<3>(tuple));
					}
					{
						dilations.emplace_back(std::get<4>(tuple));
						dilations.emplace_back(std::get<5>(tuple));
					}
					nodeInfo = {common::OpType::Conv,
						{{"group", static_cast<graph::Int>group},
						{"kernel_s"}}}
			}
			CASE(Relu)
			CASE(Add)
			default :
				IT_TODO_HALT_MSG("Don't Support OpType");
	}
#undef CASE
	return nodeInfo;
}
} // namespace infini
