#pragma once
#include "core/common.h"
#include "core/graph.h"
#include "core/tensor.h"
#include "frontend/graph.h"
#include "utils/operator_utils.h"

namespace infini {
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
            auto [it, ok] = _caches.try_emplace(i, nullptr);
            if (!ok) {
                map->setDataBlob(it->second);
                continue;
            }
            auto const &tensor = edges[i].tensor;
            auto ptr = runtime->alloc(tensor->bytesSize());
            it->second = make_ref<BlobObj>(runtime, ptr);
            map->setDataBlob(it->second);
            map->copyin(tensor->data->get<void>(), tensor->bytesSize());
        }

        return {inputs, outputs, std::move(obj)};
    }
};

} // namespace infini
