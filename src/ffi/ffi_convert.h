#pragma once

#include "core/common.h"
#include "core/graph.h"
#include "frontend/graph.h"
#include "operators/all_gather.h"
#include "operators/all_reduce.h"
#include "operators/batch_norm.h"
#include "operators/concat.h"
#include "operators/conv.h"
#include "operators/element_wise.h"
#include "operators/expand.h"
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
#include "operators/where.h"

namespace infini {

void addOperatorFromGraphTopo(
    GraphObj &g, std::vector<refactor::frontend::Edge> const &edges,
    refactor::frontend::Operator const &op,
    refactor::common::slice_t<size_t> i_, refactor::common::range_t<size_t> o,
    std::unordered_map<size_t, Tensor> &edgeToTensor,
    std::vector<std::pair<size_t, Tensor>> &weights) {
    auto fn = [&edgeToTensor, &edges, &weights, &g](size_t edgeIdx) -> Tensor {
        auto it = edgeToTensor.find(edgeIdx);
        if (it != edgeToTensor.end()) {
            return it->second;
        }
        auto const &tensor = edges[edgeIdx].tensor;
        Shape shape(tensor->shape.size());
        std::transform(tensor->shape.begin(), tensor->shape.end(),
                       shape.begin(),
                       [](auto const &ele) { return ele.value(); });
        auto tensor_ =
            g.addTensor(std::move(shape), DataType(tensor->dataType.internal));
        if (tensor->hasData()) {
            tensor_->setExternal();
            weights.emplace_back(edgeIdx, tensor_);
        }
        edgeToTensor.insert({edgeIdx, tensor_});
        return tensor_;
    };

    auto name = op.opType.name();
    if (name == "onnx::Reshape") {
        auto const &shape = *edges[i_[1]].tensor;
        auto shapeValue = shape.data->get<int64_t>();
        Shape shape_(shape.shape[0].value());
        std::transform(shapeValue, shapeValue + shape_.size(), shape_.begin(),
                       [](auto const &ele) { return static_cast<int>(ele); });
        g.addOpWithOutputs<ReshapeObj>(fn(i_[0]), fn(o[0]), std::move(shape_));
    } else if (name == "onnx::Expand") {
        auto const &shape = *edges[i_[1]].tensor;
        auto shapeValue = shape.data->get<int64_t>();
        Shape shape_(shape.shape[0].value());
        std::transform(shapeValue, shapeValue + shape_.size(), shape_.begin(),
                       [](auto const &ele) { return static_cast<int>(ele); });
        g.addOpWithOutputs<ExpandObj>(fn(i_[0]), fn(o[0]), std::move(shape_));
    } else if (name == "onnx::Unsqueeze") {
        using refactor::common::slice;
        auto const &data = *edges[i_[0]].tensor;
        auto const &axes = *edges[i_[1]].tensor;

        auto axesValue = axes.data->get<int64_t>();
        auto axesSize = axes.shape[0].value();
        auto rank = data.rank() + axesSize;

        Shape shape(data.shape.size());
        std::transform(
            data.shape.begin(), data.shape.end(), shape.begin(),
            [](auto const &ele) { return static_cast<int>(ele.value()); });
        shape.reserve(rank);

        for (auto axis : slice(axesValue, axesSize)) {
            if (axis < 0) {
                axis += rank;
            }
            shape.insert(shape.begin() + axis, 1);
        }
        g.addOpWithOutputs<ReshapeObj>(fn(i_[0]), fn(o[0]), std::move(shape));
    } else if (name == "onnx::Add") {
        g.addOpWithOutputs<AddObj>(fn(i_[0]), fn(i_[1]), fn(o[0]));
    } else if (name == "onnx::Sub") {
        g.addOpWithOutputs<SubObj>(fn(i_[0]), fn(i_[1]), fn(o[0]));
    } else if (name == "onnx::Mul") {
        g.addOpWithOutputs<MulObj>(fn(i_[0]), fn(i_[1]), fn(o[0]));
    } else if (name == "onnx::Div") {
        g.addOpWithOutputs<DivObj>(fn(i_[0]), fn(i_[1]), fn(o[0]));
    } else if (name == "onnx::Pow") {
        g.addOpWithOutputs<PowerObj>(fn(i_[0]), fn(i_[1]), fn(o[0]));
    } else if (name == "onnx::Gather") {
        auto axis = op.attribute("axis", {0}).int_();
        g.addOpWithOutputs<GatherObj>(fn(i_[0]), fn(i_[1]), fn(o[0]), axis);
    } else if (name == "onnx::Max") {
        g.addOpWithOutputs<MaximumObj>(fn(i_[0]), fn(i_[1]), fn(o[0]));
    } else if (name == "onnx::Slice") {
        using refactor::common::range0_;

        auto const &data = *edges[i_[0]].tensor;
        auto const &starts__ = *edges[i_[1]].tensor;
        auto const &ends__ = *edges[i_[2]].tensor;

        // clang-format off
            int64_t const
            *starts = starts__.data->get<int64_t>(),
            *ends   = ends__  .data->get<int64_t>(),
            *axes   = i_.size() >= 4 ? edges[i_[3]].tensor->data->get<int64_t>() : nullptr,
            *steps  = i_.size() == 5 ? edges[i_[4]].tensor->data->get<int64_t>() : nullptr;
        // clang-format on

        auto rank = data.rank();
        auto size = starts__.shape[0].value();

        Shape starts_(size), ends_(size), axes_(size), steps_(size);

        for (auto i_ : range0_(size)) {
            int64_t axis = axes ? axes[i_] : i_, step = steps ? steps[i_] : 1,
                    start = starts[i_], end = ends[i_];

            if (axis < 0) {
                axis += rank;
            }

            auto dim = data.shape[axis].value();
            if (start < 0) {
                start += dim;
            }
            if (end < 0) {
                end += dim;
            }

            starts_[i_] = static_cast<int>(std::clamp(start, 0l, dim));
            ends_[i_] = static_cast<int>(std::clamp(end, 0l, dim));
            axes_[i_] = static_cast<int>(axis);
            steps_[i_] = static_cast<int>(step);
        }
        g.addOpWithOutputs<SliceObj>(fn(i_[0]), fn(o[0]), std::move(starts_),
                                     std::move(ends_), std::move(axes_),
                                     std::move(steps_));

    } else if (name == "onnx::Softmax") {
        auto axis = op.attribute("axis", {-1}).int_();
        g.addOpWithOutputs<SoftmaxObj>(fn(i_[0]), fn(o[0]), axis);
    } else if (name == "onnx::ReduceMean") {
        auto keepdims = op.attribute("keepdims", {1}).int_();
        std::optional<std::vector<int>> axes;
        if (i_.size() > 1) {
            auto const axes_ = *edges[i_[1]].tensor;
            auto axesValue = axes_.data->get<int64_t>();
            auto axesRank = axes_.shape[0].value();
            *axes = std::vector<int>(axesRank);
            std::transform(
                axesValue, axesValue + axesRank, axes->begin(),
                [](auto const &ele) { return static_cast<int>(ele); });
        }
        g.addOpWithOutputs<ReduceMeanObj>(fn(i_[0]), fn(o[0]), std::move(axes),
                                          keepdims);
    } else if (name == "onnx::Concat") {
        auto axis = op.attribute("axis").int_();
        std::vector<Tensor> inputs(i_.size());
        std::transform(i_.begin(), i_.end(), inputs.begin(), fn);
        g.addOpWithOutputs<ConcatObj>(std::move(inputs), fn(o[0]), axis);
    } else if (name == "onnx::MatMul") {
        g.addOpWithOutputs<MatmulObj>(fn(i_[0]), fn(i_[1]), fn(o[0]), false,
                                      false, nullptr, ActType::None);
    } else if (name == "onnx::Gemm") {
        auto alpha = op.attribute("alpha", {1.0f}).float_();
        auto beta = op.attribute("beta", {1.0f}).float_();
        auto transA = op.attribute("transA", {0}).int_() != 0;
        auto transB = op.attribute("transB", {0}).int_() != 0;
        IT_ASSERT(alpha == 1.0);
        IT_ASSERT(beta == 1.0);
        g.addOpWithOutputs<MatmulObj>(
            fn(i_[0]), fn(i_[1]), fn(o[0]), transA, transB,
            i_.size() > 2 ? fn(i_[2]) : nullptr, ActType::None);
    } else if (name == "onnx::Transpose") {
        auto const &data = *edges[i_[0]].tensor;
        auto const &attrs = op.attributes;

        if (auto it = attrs.find("perm"); it != attrs.end()) {
            auto const &perm = it->second.ints();
            Shape perm_(perm.size());
            std::transform(
                perm.begin(), perm.end(), perm_.begin(),
                [](auto const &ele) { return static_cast<int>(ele); });
            g.addOpWithOutputs<TransposeObj>(fn(i_[0]), fn(o[0]),
                                             std::move(perm_));
        } else {
            using refactor::common::range0_;
            Shape perm(data.rank());
            auto num = range0_(data.rank());
            std::copy(num.begin(), num.end(), perm.rbegin());
            g.addOpWithOutputs<TransposeObj>(fn(i_[0]), fn(o[0]),
                                             std::move(perm));
        }
    } else if (name == "onnx::Split") {
        auto axis = op.attribute("axis", {0}).int_();
        if (axis < 0) {
            axis += edges[i_[0]].tensor->rank();
        }
        std::vector<Tensor> outputs(o.size());
        std::transform(o.begin(), o.end(), outputs.begin(), fn);
        int num = o.size();
        if (i_.size() == 1) {
            g.addOpWithOutputs<SplitObj>(fn(i_[0]), std::move(outputs), axis,
                                         num);
        } else {
            auto ratioValue = edges[i_[1]].tensor->data->get<int64_t>();
            auto rank = edges[i_[1]].tensor->shape[0].value();
            std::vector<int> ratio(rank);
            std::transform(
                ratioValue, ratioValue + rank, ratio.begin(),
                [](auto const &ele) { return static_cast<int>(ele); });
            g.addOpWithOutputs<SplitObj>(fn(i_[0]), std::move(outputs), axis,
                                         std::move(ratio));
        }
    } else if (name == "onnx::Where") {
        g.addOpWithOutputs<WhereObj>(fn(i_[1]), fn(i_[2]), fn(i_[0]), fn(o[0]));
    } else if (name == "onnx::Softmax") {
        auto axis = op.attribute("axis", {-1}).int_();
        g.addOpWithOutputs<SoftmaxObj>(fn(i_[0]), fn(o[0]), axis);
    } else if (name == "onnx::Sqrt") {
        g.addOpWithOutputs<SqrtObj>(fn(i_[0]), fn(o[0]));
    } else if (name == "onnx::Relu") {
        g.addOpWithOutputs<ReluObj>(fn(i_[0]), fn(o[0]));
    } else if (name == "onnx::Identity") {
        g.addOpWithOutputs<IdentityObj>(fn(i_[0]), fn(o[0]));
    } else if (name == "onnx::Tanh") {
        g.addOpWithOutputs<TanhObj>(fn(i_[0]), fn(o[0]));
    } else if (name == "onnx::AllReduceSum") {
        g.addOpWithOutputs<AllReduceSumObj>(fn(i_[0]), fn(o[0]));
    } else if (name == "onnx::AllReduceProd") {
        g.addOpWithOutputs<AllReduceProdObj>(fn(i_[0]), fn(o[0]));
    } else if (name == "onnx::AllReduceMin") {
        g.addOpWithOutputs<AllReduceMinObj>(fn(i_[0]), fn(o[0]));
    } else if (name == "onnx::AllReduceMax") {
        g.addOpWithOutputs<AllReduceMaxObj>(fn(i_[0]), fn(o[0]));
    } else if (name == "onnx::AllReduceAvg") {
        g.addOpWithOutputs<AllReduceAvgObj>(fn(i_[0]), fn(o[0]));
    } else if (name == "onnx::AllGather") {
        auto size = o.size();
        std::vector<Tensor> outputs(size);
        std::transform(o.begin(), o.end(), outputs.begin(), fn);
        g.addOpWithOutputs<AllGatherObj>(fn(i_[0]), std::move(outputs), size);
    } else if (name == "onnx::Less") {
        g.addOpWithOutputs<LessThanObj>(fn(i_[0]), fn(i_[1]), fn(o[0]));
    } else {
        std::cerr << "Unknown operator: " << name << std::endl;
        IT_ASSERT_TODO("");
    }
}

struct ConvertGraphObj {
    std::unordered_map<size_t, Blob> _caches;

    std::tuple<std::vector<Tensor>, std::vector<Tensor>, Graph>
    convert(refactor::frontend::Graph const &graph, Runtime runtime) {
        auto obj = make_ref<GraphObj>(runtime);
        auto const &nodes = graph.internal().nodes;
        auto const &edges = graph.internal().edges;
        std::unordered_map<size_t, Tensor> edgeToTensor;
        std::vector<std::pair<size_t, Tensor>> weights;

        auto it = graph.internal().topology.begin(),
             end = graph.internal().topology.end();
        while (it != end) {
            auto [nodeIdx, i, o] = *it++;
            // not dynamic_node
            if (std::any_of(o.begin(), o.end(), [&edges](auto e) {
                    return !edges[e].tensor->hasData();
                })) {
                addOperatorFromGraphTopo(*obj, edges, nodes[nodeIdx].op, i, o,
                                         edgeToTensor, weights);
            }
        }
        std::vector<Tensor> inputs, outputs;
        inputs.reserve(it.globalInputs().size());
        outputs.reserve(it.globalOutputs().size());
        for (auto edgeIdx : it.globalInputs()) {
            auto tensor = edgeToTensor.at(edgeIdx);
            tensor->setInput();
            inputs.emplace_back(tensor);
        }
        for (auto edgeIdx : it.globalOutputs()) {
            if (auto it_ = edgeToTensor.find(edgeIdx);
                it_ != edgeToTensor.end()) {
                it_->second->setOutput();
                outputs.emplace_back(it_->second);
            } else {
                outputs.emplace_back(nullptr);
            }
        }

        obj->dataMalloc();
        for (auto &[i, map] : weights) {
            if (auto [it_, ok] = _caches.try_emplace(i, nullptr); ok) {
                auto const &tensor = edges[i].tensor;
                auto ptr = runtime->alloc(tensor->bytesSize());
                map->setDataBlob(it_->second = make_ref<BlobObj>(runtime, ptr));
                map->copyin(tensor->data->get<void>(), tensor->bytesSize());
            } else {
                map->setDataBlob(it_->second);
            }
        }

        return {inputs, outputs, std::move(obj)};
    }
};

} // namespace infini
