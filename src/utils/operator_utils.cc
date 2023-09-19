#include "utils/operator_utils.h"
#include "operators/batch_norm.h"
#include "operators/concat.h"
#include "operators/conv.h"
#include "operators/element_wise.h"
#include "operators/gather.h"
#include "operators/matmul.h"
#include "operators/pad.h"
#include "operators/pooling.h"
#include "operators/reduce_mean.h"
#include "operators/reshape.h"
#include "operators/slice.h"
#include "operators/softmax.h"
#include "operators/split.h"
#include "operators/transpose.h"
#include "operators/unary.h"

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

void addOperatorFromGraphTopo(
    GraphObj &g, std::shared_ptr<refactor::computation::Operator> nodeInfo,
    std::vector<size_t> input, std::vector<size_t> output,
    std::unordered_map<size_t, Tensor> &edgeToTensor,
    std::vector<refactor::computation::Edge> edges) {
    std::string name(nodeInfo->opType.name());
    auto attr = nodeInfo->attributes;
    if (name == "onnx::Conv") {
        //	auto p = attr["pads"].ints();
        //	auto s = attr["strides"].ints();
        //	auto d = attr["dilations"].ints();
        //	g.addOpWithOutputs<ConvObj>(edgeToTensor[input[0]],
        // edgeToTensor[input[1]], edgeToTensor[output[0]], p[0], p[1], s[0],
        // s[1], d[0], d[1]);
    } else if (name == "onnx::Add") {
        g.addOpWithOutputs<AddObj>(edgeToTensor[input[0]],
                                   edgeToTensor[input[1]],
                                   edgeToTensor[output[0]]);
    } else if (name == "onnx::AveragePool") {
        //	auto p = attr["pads"].ints();
        //	auto s = attr["strides"].ints();
        //	auto d = attr["dilations"].ints();
        //	int h = edgeToTensor[input[0]]->getDims()[2];
        //	int w = edgeToTensor[input[0]]->getDims()[3];
        //	g.addOpWithOutputs<AvgPoolObj>(edgeToTensor[input[0]],
        // edgeToTensor[output[0]], h, w,
        //                                   d[0], d[1], p[0], p[1], s[0],
        //                                   s[1]);
    } else if (name == "onnx::Reshape") {
        IT_ASSERT(input.size() == 2);
        auto shapeValue =
            reinterpret_cast<int64_t *>(edges[input[1]].tensor->data->ptr);
        auto rank = edgeToTensor[input[1]]->getDims()[0];
        Shape shape(rank);
        for (size_t i = 0; i < (size_t)rank; ++i) {
            shape[i] = static_cast<int>(*(shapeValue + i));
        }
        g.addOpWithOutputs<ReshapeObj>(edgeToTensor[input[0]],
                                       edgeToTensor[output[0]], shape);
    } else if (name == "onnx::Gemm") {
        auto alpha =
            attr.find("alpha") != attr.end() ? attr["alpha"].float_() : 1.0;
        auto beta =
            attr.find("beta") != attr.end() ? attr["beta"].float_() : 1.0;
        auto transA =
            attr.find("transA") != attr.end() ? attr["transA"].int_() : 0;
        auto transB =
            attr.find("transB") != attr.end() ? attr["transB"].int_() : 0;
        IT_ASSERT(alpha == 1.0);
        IT_ASSERT(beta == 1.0);
        g.addOpWithOutputs<MatmulObj>(
            edgeToTensor[input[0]], edgeToTensor[input[1]],
            edgeToTensor[output[0]], transA, transB,
            input.size() > 2 ? edgeToTensor[input[2]] : nullptr, ActType::None);
    } else if (name == "onnx::Pow") {
        g.addOpWithOutputs<PowerObj>(edgeToTensor[input[0]],
                                     edgeToTensor[input[1]],
                                     edgeToTensor[output[0]]);
    } else if (name == "onnx::Gather") {
        auto axis = attr.find("axis") != attr.end() ? attr["axis"].int_() : 0;
        g.addOpWithOutputs<GatherObj>(edgeToTensor[input[0]],
                                      edgeToTensor[input[1]],
                                      edgeToTensor[output[0]], axis);
    } else if (name == "onnx::Max") {
        g.addOpWithOutputs<MaximumObj>(edgeToTensor[input[0]],
                                       edgeToTensor[input[1]],
                                       edgeToTensor[output[0]]);
    } else if (name == "onnx::Div") {
        g.addOpWithOutputs<DivObj>(edgeToTensor[input[0]],
                                   edgeToTensor[input[1]],
                                   edgeToTensor[output[0]]);
    } else if (name == "onnx::Mul") {
        g.addOpWithOutputs<MulObj>(edgeToTensor[input[0]],
                                   edgeToTensor[input[1]],
                                   edgeToTensor[output[0]]);
    } else if (name == "onnx::Sub") {
        g.addOpWithOutputs<SubObj>(edgeToTensor[input[0]],
                                   edgeToTensor[input[1]],
                                   edgeToTensor[output[0]]);
    } else if (name == "onnx::Slice") {
        auto startValue =
            reinterpret_cast<int64_t *>(edges[input[1]].tensor->data->ptr);
        auto startRank = edgeToTensor[input[1]]->getRank();
        auto endValue =
            reinterpret_cast<int64_t *>(edges[input[2]].tensor->data->ptr);
        auto endRank = edgeToTensor[input[2]]->getRank();
        std::vector<int> start, end, axesVal, stepsVal;
        std::optional<std::vector<int>> axes, steps;
        if (input.size() > 3) {
            auto axesValue =
                reinterpret_cast<int64_t *>(edges[input[3]].tensor->data->ptr);
            auto axesRank = edgeToTensor[input[3]]->getRank();
            for (size_t i = 0; i < axesRank; ++i) {
                axesVal.emplace_back(static_cast<int>(*(axesValue + i)));
            }
            axes = axesVal;
        }
        if (input.size() > 4) {
            auto stepsValue =
                reinterpret_cast<int64_t *>(edges[input[4]].tensor->data->ptr);
            auto stepsRank = edgeToTensor[input[4]]->getRank();
            for (size_t i = 0; i < stepsRank; ++i) {
                stepsVal.emplace_back(static_cast<int>(*(stepsValue + i)));
            }
            steps = stepsVal;
        }
        for (size_t i = 0; i < startRank; ++i) {
            int64_t startVal = *(startValue + i);
            if (axes.has_value()) {
                startVal = std::min(
                    startVal,
                    static_cast<int64_t>(
                        edgeToTensor[input[0]]->getDims()[axes.value()[i]]));
            } else {
                startVal = std::min(
                    startVal,
                    static_cast<int64_t>(edgeToTensor[input[0]]->getDims()[i]));
            }
            start.emplace_back(static_cast<int>(startVal));
        }
        for (size_t i = 0; i < endRank; ++i) {
            int64_t endVal = *(endValue + i);
            if (axes.has_value()) {
                endVal = std::min(
                    endVal,
                    static_cast<int64_t>(
                        edgeToTensor[input[0]]->getDims()[axes.value()[i]]));
            } else {
                endVal = std::min(
                    endVal,
                    static_cast<int64_t>(edgeToTensor[input[0]]->getDims()[i]));
            }
            end.emplace_back(static_cast<int>(endVal));
        }
        g.addOpWithOutputs<SliceObj>(edgeToTensor[input[0]],
                                     edgeToTensor[output[0]], start, end, axes,
                                     steps);
    } else if (name == "onnx::Softmax") {
        auto axis = attr.find("axis") != attr.end() ? attr["axis"].int_() : -1;
        g.addOpWithOutputs<SoftmaxObj>(edgeToTensor[input[0]],
                                       edgeToTensor[output[0]], axis);
    } else if (name == "onnx::ReduceMean") {
        auto keepdims =
            attr.find("keepdims") != attr.end() ? attr["keepdims"].int_() : 1;
        std::vector<int> axesVal;
        std::optional<std::vector<int>> axes;
        if (input.size() > 1) {
            auto axesValue =
                reinterpret_cast<int64_t *>(edges[input[1]].tensor->data->ptr);
            auto axesRank = edgeToTensor[input[1]]->getRank();
            for (size_t i = 0; i < axesRank; ++i) {
                axesVal.emplace_back(static_cast<int>(*(axesValue + i)));
            }
            axes = axesVal;
        }
        g.addOpWithOutputs<ReduceMeanObj>(
            edgeToTensor[input[0]], edgeToTensor[output[0]], axes, keepdims);
    } else if (name == "onnx::Concat") {
        auto axis = attr["axis"].int_();
        std::vector<Tensor> inputs;
        for (auto i : input) {
            inputs.emplace_back(edgeToTensor[i]);
        }
        g.addOpWithOutputs<ConcatObj>(inputs, edgeToTensor[output[0]], axis);
    } else if (name == "onnx::MatMul") {
        g.addOpWithOutputs<MatmulObj>(
            edgeToTensor[input[0]], edgeToTensor[input[1]],
            edgeToTensor[output[0]], false, false, nullptr, ActType::None);
    } else if (name == "onnx::Transpose") {
        int rank = edgeToTensor[input[0]]->getRank();
        std::vector<int> permDefault;
        for (int i = rank - 1; i >= 0; --i) {
            permDefault.emplace_back(i);
        }
        std::vector<int> perm;
        if (attr.find("perm") != attr.end()) {
            auto permAttr = attr["perm"].ints();
            for (auto e : permAttr) {
                perm.emplace_back(static_cast<int>(e));
            }
        } else {
            perm = permDefault;
        }
        g.addOpWithOutputs<TransposeObj>(edgeToTensor[input[0]],
                                         edgeToTensor[output[0]], perm);
    } else if (name == "onnx::Split") {
		auto axis = attr.find("axis") != attr.end() ? attr["axis"].int_() : 0;
		std::vector<Tensor> outputs;
		for (auto i : output) {
			outputs.emplace_back(edgeToTensor[i]);
		}
		int num = output.size();
		if (input.size() == 2) {
			auto ratioValue = reinterpret_cast<int64_t *>(edges[input[1]].tensor->data->ptr);
			std::vector<int> ratio;
			auto rank = edgeToTensor[input[1]]->getDims()[0];
			for (size_t i = 0; i < (size_t)rank; ++i) {
				ratio.emplace_back(static_cast<int>(*(ratioValue + i)));
			}
			g.addOpWithOutputs<SplitObj>(edgeToTensor[input[0]], outputs, axis, ratio);
		} else {
			g.addOpWithOutputs<SplitObj>(edgeToTensor[input[0]], outputs, axis, num);
		}
	} else if (name == "onnx::Where") {
		IT_ASSERT(input.size() == 3);
		g.addOpWithOutputs<WhereObj>(edgeToTensor[input[1]], edgeToTensor[input[2]],
									edgeToTensor[input[0]], edgeToTensor[output[0]]);
	} else if (name == "onnx::Softmax") {
		//auto axis = attr.find("axis") != attr.end() ? attr["axis"].int_() : -1;
		
	} else if (name == "onnx::Sqrt") {
        g.addOpWithOutputs<SqrtObj>(edgeToTensor[input[0]], 
                                    edgeToTensor[output[0]]); 
	} else if (name == "onnx::Relu") {
        g.addOpWithOutputs<ReluObj>(edgeToTensor[input[0]], 
                                    edgeToTensor[output[0]]); 
	} else if (name == "onnx::Identity") {
        g.addOpWithOutputs<IdentityObj>(edgeToTensor[input[0]], 
                                    edgeToTensor[output[0]]); 
	} else if (name == "onnx::Tanh") {
        g.addOpWithOutputs<TanhObj>(edgeToTensor[input[0]], 
                                    edgeToTensor[output[0]]); 
	}
}

void addEdgeToTensor(GraphObj &g, size_t index,
                     std::shared_ptr<refactor::computation::Tensor> tensor,
                     std::unordered_map<size_t, Tensor> &edgeToTensor,
                     Runtime runtime) {
    auto refShape = tensor->shape;
    Shape shape;
    for (auto ele : refShape) {
        IT_ASSERT(ele.hasValue());
        shape.emplace_back(ele.value());
    }
    auto dType = tensor->dataType;
    Tensor tensorInf = g.addTensor(shape, DataType(static_cast<int>(dType)));
    edgeToTensor.insert(std::make_pair(index, tensorInf));
}
} // namespace infini
