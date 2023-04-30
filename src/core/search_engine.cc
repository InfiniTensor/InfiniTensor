#include "core/search_engine.h"
#include "core/hash.h"
#include "core/runtime.h"
#include "ffi/ffi_callback.h"
#include "nnet/dbg.h"
#include "operators/reshape.h"

#include <algorithm>
#include <iostream>
#include <unordered_set>

namespace infini {

using MetaGraph = SearchEngine::MetaGraph;

SearchEngine::SearchEngine(Runtime runtime, Ref<Mutator> mutator)
    : runtimeExec(runtime), mutator(mutator) {
    // Compare graph with estimated time
    graphTimeComparer = [this](const Graph &a, const Graph &b) -> bool {
        return getEstimatedGraphPerf(a) < getEstimatedGraphPerf(b);
    };
}

void SearchEngine::printMetaGraph(MetaGraph metaGraph) {
    for (size_t i = 0; i < metaGraph->nodes.size(); i++) {
        auto &node = metaGraph->nodes[i];
        std::cout << "id: " << i << std::endl;
        node.graph->print();
        std::cout << "type: " << node.type << std::endl;
        std::cout << "pre: ";
        for (auto &x : node.pre) {
            std::cout << x << " ";
        }
        std::cout << std::endl;
        std::cout << "suc: ";
        for (auto &x : node.suc) {
            std::cout << x << " ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
}

Graph SearchEngine::run(const Graph graph) {
    IT_ASSERT(runtimeExec == graph->getRuntime());
    std::cout << "[INFO] original graph: " << std::endl;
    std::cout << graph->toString();
    std::cout << "[INFO] perf: " << getEstimatedGraphPerf(graph) << std::endl;

    std::vector<Graph> partitions = partitionGraph(graph);

    std::cout << "[INFO] Partition num: " << partitions.size() << std::endl;
    std::vector<Graph> bestGraphs = {nullptr};
    for (size_t pid = 0; pid < partitions.size(); pid++) {
        auto &subGraph = partitions[pid];
        std::cout << "[INFO] Partition: " << pid << std::endl;
        std::vector<Graph> candidates = search(subGraph);
        std::cout << "[INFO] size: " << candidates.size() << std::endl;
        IT_ASSERT(candidates.size() > 0);
        std::vector<Graph> nextGraphs;
        for (auto lastGraph : bestGraphs) {
            for (auto thisGraph : candidates) {
                std::vector<Operator> ops;
                if (lastGraph != nullptr) {
                    for (auto op : lastGraph->getOperators()) {
                        ops.emplace_back(op);
                    }
                }
                if (thisGraph != nullptr) {
                    for (auto op : thisGraph->getOperators()) {
                        ops.emplace_back(op);
                    }
                }
                auto tmp = make_ref<GraphObj>(runtimeExec, ops);
                nextGraphs.emplace_back(tmp);
            }
        }
        dbg("===Num" + std::to_string(nextGraphs.size()));
        std::sort(nextGraphs.begin(), nextGraphs.end(), graphTimeComparer);

        if (nextGraphs.size() > GRAPH_SIZE) {
            nextGraphs.resize(GRAPH_SIZE);
        }
        bestGraphs.clear();
        for (size_t i = 0; i < nextGraphs.size(); i++) {
            bestGraphs.emplace_back(nextGraphs[i]);
        }
    }

    std::cout << "[INFO] unfused graph: " << std::endl;
    for (size_t i = 0; i < bestGraphs.size(); i++) {
        std::cout << "bestGraph " << i << ":" << std::endl;
        std::cout << bestGraphs[i]->toString();
        std::cout << "[INFO] perf: " << getEstimatedGraphPerf(bestGraphs[i])
                  << std::endl;
    }

    // Fuse vertically and sort according to performance
    for (size_t i = 0; i < bestGraphs.size(); ++i) {
        bestGraphs[i] = fuseVertically(bestGraphs[i]);
    }
    std::sort(bestGraphs.begin(), bestGraphs.end(), graphTimeComparer);

    // Check optimized graphs are legal
    for (auto g : bestGraphs) {
        g->checkValid();
        IT_ASSERT(graph->getInputs().size() == g->getInputs().size(),
                  graph->toString() + string("\n") + g->toString());
        IT_ASSERT(graph->getOutputs().size() == g->getOutputs().size(),
                  graph->toString() + string("\n") + g->toString());
    }

    std::cout << "[INFO] best fused graph: " << std::endl;
    std::cout << "[INFO] perf: " << getEstimatedGraphPerf(bestGraphs[0])
              << std::endl;
    std::cout << bestGraphs[0] << std::endl;

    return bestGraphs[0];
}

std::vector<Graph> SearchEngine::search(const Graph &graph) {
    auto metaGraph = buildMetaGraphWithGraph(graph);
    auto mergedGraphs = searchMerge(metaGraph);
    std::cout << "[INFO] merged graphs: " << mergedGraphs.size() << std::endl;

    std::vector<Graph> results;
    for (auto mergedGraph : mergedGraphs) {
        auto mutatedGraphs = searchMutation(mergedGraph);
        for (size_t i = 0; i < std::min(mutatedGraphs.size(), GRAPH_SIZE);
             i++) {
            results.emplace_back(mutatedGraphs[i]);
        }
    }

    // compare with perf time
    dbg("===Num" + std::to_string(results.size()));
    std::sort(results.begin(), results.end(), graphTimeComparer);
    if (results.size() > GRAPH_SIZE) {
        results.resize(GRAPH_SIZE);
    }
    return results;
}

// Build metagraph with a graph, each operator is a node.
MetaGraph SearchEngine::buildMetaGraphWithGraph(const Graph graph) {
    auto metaGraph = make_ref<MetaGraphObj>();

    int numOps = graph->getOperators().size();
    std::vector<int> cnt(numOps, 0);
    std::unordered_map<int, int> opMap;
    metaGraph->nodes.clear();
    std::vector<int> q(0);
    for (size_t i = 0; i < graph->getOperators().size(); i++) {
        auto &op = graph->getOperators()[i];
        MetaGraphObj::Node node;
        std::vector<Operator> ops;
        ops.emplace_back(op);
        node.graph = make_ref<GraphObj>(runtimeExec, ops);
        node.type = op->isComputeOp();
        node.cnt = op->getPredecessors().size();
        opMap.emplace(op->getGuid(), i);
        metaGraph->nodes.emplace_back(node);
    }
    for (size_t i = 0; i < graph->getOperators().size(); i++) {
        auto &op = graph->getOperators()[i];
        std::unordered_set<int> set;
        set.clear();
        set.emplace(i);
        for (auto preOp : op->getPredecessors()) {
            int id = opMap[preOp->getGuid()];
            if (set.find(id) == set.end()) {
                metaGraph->nodes[i].pre.emplace_back(id);
                set.emplace(id);
            }
        }
        for (auto sucOp : op->getSuccessors()) {
            int id = opMap[sucOp->getGuid()];
            if (set.find(id) == set.end()) {
                metaGraph->nodes[i].suc.emplace_back(id);
                set.emplace(id);
            }
        }
    }
    return metaGraph;
}

// Build a metagraph with graph and a plan, a plan is which ops should be a
// node.
MetaGraph SearchEngine::buildMetaGraphWithPlan(const MetaGraph metaGraph,
                                               const std::vector<int> &plan) {
    int numGroups = 0;
    for (auto i : plan) {
        if (i > numGroups) {
            numGroups = i;
        }
    }

    std::vector<std::vector<int>> groups(numGroups + 1, std::vector<int>(0));
    for (size_t i = 0; i < plan.size(); i++) {
        groups[plan[i]].emplace_back(i);
    }

    auto resultMetaGraph = make_ref<MetaGraphObj>();
    for (auto &group : groups) {
        std::vector<Operator> ops;
        std::unordered_set<int> preSet, sucSet;
        for (auto id : group) {
            MetaGraphObj::Node node;
            for (auto op : metaGraph->nodes[id].graph->getOperators()) {
                ops.emplace_back(op);
            }
            for (auto suc : metaGraph->nodes[id].suc) {
                if (sucSet.find(plan[suc]) == sucSet.end()) {
                    node.suc.emplace_back(plan[suc]);
                    sucSet.emplace(plan[suc]);
                }
            }
            for (auto pre : metaGraph->nodes[id].pre) {
                IT_ASSERT(sucSet.find(plan[pre]) == sucSet.end());
                if (preSet.find(plan[pre]) == preSet.end()) {
                    node.pre.emplace_back(plan[pre]);
                    preSet.emplace(plan[pre]);
                }
            }
            node.graph = make_ref<GraphObj>(runtimeExec, ops);
            node.cnt = node.pre.size();
            node.type = ops[0]->isComputeOp();
            resultMetaGraph->nodes.emplace_back(node);
        }
    }
    return resultMetaGraph;
}

// Search how to merge multiple ops.
vector<MetaGraph> SearchEngine::searchMerge(MetaGraph &metaGraph) {
    IT_ASSERT(metaGraph != nullptr);
    // HACK: disable multiple op search
    return {metaGraph};
    std::vector<int> plan(metaGraph->nodes.size());
    for (size_t i = 0; i < plan.size(); i++) {
        plan[i] = i;
    }
    std::vector<int> frontier;
    for (size_t i = 0; i < plan.size(); i++) {
        if (metaGraph->nodes[i].cnt == 0) {
            frontier.emplace_back(i);
        }
    }

    std::vector<std::vector<int>> plans;
    std::unordered_set<HashType> planSet;
    searchMergeDfs(metaGraph, plan, frontier, plans, planSet);

    vector<MetaGraph> metaGraphs;
    for (auto &curPlan : plans) {
        metaGraphs.emplace_back(buildMetaGraphWithPlan(metaGraph, curPlan));
    }
    return metaGraphs;
}

// DFS impl for search merge.
void SearchEngine::searchMergeDfs(MetaGraph &metaGraph, std::vector<int> &plan,
                                  std::vector<int> &frontier,
                                  std::vector<std::vector<int>> &plans,
                                  std::unordered_set<uint64_t> &planSet) {
    if (frontier.size() == 0) {
        // remark id
        std::unordered_map<int, int> id_map;
        int cnt = 0;
        for (size_t i = 0; i < plan.size(); i++) {
            if (id_map.find(plan[i]) == id_map.end()) {
                id_map.emplace(plan[i], cnt++);
            }
            plan[i] = id_map[plan[i]];
        }
        auto hash = hashVector(plan);
        if (planSet.find(hash) != planSet.end()) {
            return;
        }
        planSet.emplace(hash);
        plans.emplace_back(plan);
        return;
    }

    int numNonCompute = 0;
    for (auto x : frontier) {
        if (metaGraph->nodes[x].type == 0) {
            numNonCompute++;
        }
    }

    auto planBackup = plan;
    auto metaGraphBackup = metaGraph;
    // DFS non compute ops.
    if (numNonCompute > 0) {
        std::vector<int> nextFrontier;
        for (auto x : frontier) {
            if (metaGraph->nodes[x].type == 0) {
                for (auto y : metaGraph->nodes[x].suc) {
                    metaGraph->nodes[y].cnt--;
                    if (metaGraph->nodes[y].cnt == 0) {
                        nextFrontier.emplace_back(y);
                    }
                }
            } else {
                nextFrontier.emplace_back(x);
            }
        }
        searchMergeDfs(metaGraph, plan, nextFrontier, plans, planSet);
        metaGraph = metaGraphBackup;
        return;
    }

    // DFS compute ops.
    for (int mask = (1 << frontier.size()) - 1; mask > 0; mask--) {
        int mergedId = -1;
        std::vector<int> nextFrontier;
        std::vector<Operator> ops;
        for (size_t i = 0; i < frontier.size(); i++) {
            if ((1 << i) & mask) {
                if (mergedId == -1) {
                    mergedId = plan[frontier[i]];
                } else {
                    plan[frontier[i]] = mergedId;
                }
                for (auto y : metaGraph->nodes[frontier[i]].suc) {
                    metaGraph->nodes[y].cnt--;
                    if (metaGraph->nodes[y].cnt == 0) {
                        nextFrontier.emplace_back(y);
                    }
                }
                for (auto op :
                     metaGraph->nodes[frontier[i]].graph->getOperators()) {
                    ops.emplace_back(op);
                }
            } else {
                nextFrontier.emplace_back(frontier[i]);
            }
        }
        auto graph = make_ref<GraphObj>(runtimeExec, ops);
        if (ops.size() == 1 || isMultiBranchMergable(graph)) {
            searchMergeDfs(metaGraph, plan, nextFrontier, plans, planSet);
        }
        plan = planBackup;
        metaGraph = metaGraphBackup;
    }
    return;
}

// Search mutation for each compute op.
std::vector<Graph> SearchEngine::searchMutation(const MetaGraph &metaGraph) {
    std::vector<Graph> graphs = {nullptr};
    // Append a node to all existing candidates
    for (auto &node : metaGraph->nodes) {
        std::vector<Graph> nextGraphs;
        if (node.type == 1) { // If it has computing OPs
            auto mutatedGraphs = mutator->run(node.graph);
            if (mutator->hasTunedKernel)
                chooseBestMutation = false;
            if (searchFilter == 1) {
                std::sort(mutatedGraphs.begin(), mutatedGraphs.end(),
                          graphTimeComparer);
                if (mutatedGraphs.size() >= 10)
                    mutatedGraphs.resize(10);
                mutatedGraphs = {mutatedGraphs[0]};
            } else if (chooseBestMutation && mutatedGraphs.size() >= 2) {
                std::sort(mutatedGraphs.begin(), mutatedGraphs.end(),
                          graphTimeComparer);
                if (mutatedGraphs.size() >= 10)
                    mutatedGraphs.resize(10);
                mutatedGraphs = {mutatedGraphs[0]};
            } else { // avoid repeated kernel genreation
                if (mutatedGraphs.size() >= 2) // INFOGAN
                    mutatedGraphs = {mutatedGraphs[1]};
                // if (mutatedGraphs.size() > 2) {
                //     mutatedGraphs.resize(2);
                // }
            }

            for (auto graph : graphs) {
                for (auto mutatedGraph : mutatedGraphs) {
                    std::vector<Operator> ops;
                    if (graph != nullptr) {
                        for (auto op : graph->getOperators()) {
                            ops.emplace_back(op);
                        }
                    }
                    for (auto op : mutatedGraph->getOperators()) {
                        ops.emplace_back(op);
                    }
                    nextGraphs.emplace_back(
                        make_ref<GraphObj>(runtimeExec, ops));
                }
            }
        } else {
            for (auto graph : graphs) {
                std::vector<Operator> ops;
                if (graph != nullptr) {
                    for (auto op : graph->getOperators()) {
                        ops.emplace_back(op);
                    }
                }
                for (auto op : node.graph->getOperators()) {
                    ops.emplace_back(op);
                }
                nextGraphs.emplace_back(make_ref<GraphObj>(runtimeExec, ops));
            }
        }
        dbg("===Num" + std::to_string(nextGraphs.size()));
        std::sort(nextGraphs.begin(), nextGraphs.end(), graphTimeComparer);
        if (nextGraphs.size() > GRAPH_SIZE) {
            nextGraphs.resize(GRAPH_SIZE);
        }
        graphs = nextGraphs;
    }
    return graphs;
}

bool SearchEngine::isMultiBranchMergable(const Graph graph) {
    return mutator->isMultiBranchMergable(graph);
}

// Split a graph into multiple independt graphs. Search engine will search for
// each one.
std::vector<Graph> SearchEngine::partitionGraph(const Graph graph) {
    std::vector<Graph> partitions;
    // Reversed DFS post-order is topo-order.
    std::unordered_map<size_t, size_t> preOrder, postOrder;
    std::vector<Operator> ops;
    int preCnt = 0, postCnt = 0;
    std::function<void(Operator)> dfs = [&](Operator op) {
        if (preOrder.count(op->getGuid())) {
            return;
        }
        preOrder[op->getGuid()] = preCnt++;
        for (auto &&next : op->getSuccessors()) {
            dfs(next);
        }
        postOrder[op->getGuid()] = postCnt++;
        ops.emplace_back(op);
    };
    for (auto &&op : graph->getOperators()) {
        dfs(op);
    }

    std::vector<Operator> headOps;
    for (size_t i = 0; i < ops.size(); i++) {
        auto &op = ops[i];
        headOps.emplace_back(op);
        if (op->getPredecessors().size() + op->getSuccessors().size() >=
                (size_t)partitionThreshold &&
            !op->isComputeOp()) {
            auto preOrderI = preOrder[op->getGuid()];
            auto postOrderI = postOrder[op->getGuid()];
            for (size_t j = 0; j < i; j++) {
                // True predecessor
                if (preOrder[ops[j]->getGuid()] < preOrderI) {
                    for (auto nextOp : ops[j]->getSuccessors()) {
                        if (postOrder[nextOp->getGuid()] < postOrderI) {
                            // FIXME: DO NOT USE goto
                            goto fail;
                        }
                    }
                }
            }
            std::cout << "partition!!!: " << i << std::endl;
            for (auto op : headOps) {
                std::cout << op->toString() << std::endl;
            }
            auto tmp = make_ref<GraphObj>(runtimeExec, headOps);
            partitions.emplace_back(tmp);
            headOps.clear();
        }
    fail:;
    }
    if (!headOps.empty()) {
        auto tmp = make_ref<GraphObj>(runtimeExec, headOps);
        partitions.emplace_back(tmp);
    }
    std::reverse(partitions.begin(), partitions.end());
    return partitions;
}

double SearchEngine::getEstimatedGraphPerf(Graph graph) {
    // dbg(graph);
    // // hkz
    // callback::exportONNX(graph, "a.onnx");
    return runtimeExec->getPerfTime(graph, false, true, true);
}

Graph SearchEngine::fuseVertically(const Graph &graph) {
    std::unordered_map<UidBaseType, int> visitTime;
    std::vector<Operator> ops;

    graph->topo_sort();
    int cnt = 0;
    for (auto op : graph->getOperators()) {
        // Skip visited OP
        if (visitTime.find(op->getGuid()) != visitTime.end()) {
            continue;
        }
        // Skip compute OP and multi-input/output OP
        if (!op->isMemBoundOp() || (op->getPredecessors().size() != 1 &&
                                    op->getSuccessors().size() != 1)) {
            visitTime.emplace(op->getGuid(), ++cnt);
            ops.emplace_back(op);
            continue;
        }
        // FIXME: fuse and modify attributes of computing operators
        if (op->getOpType() == OpType::Relu ||
            op->getOpType() == OpType::PRelu) {
            if (auto p = op->getInputs()[0])
                if (auto sop = p->getSource())
                    if (sop->getOpType() == OpType::Conv ||
                        sop->getOpType() == OpType::Matmul) {
                        visitTime.emplace(op->getGuid(), ++cnt);
                        ops.emplace_back(make_ref<ReshapeObj>(
                            nullptr, op->getInputs()[0], op->getOutputs()[0]));
                        continue;
                    }
        }
        vector<Operator> chainOps;
        visitTime.emplace(op->getGuid(), ++cnt);

        vector<Operator> tmp;
        auto cur = op;
        while (cur->getPredecessors().size() == 1 &&
               cur->getPredecessors()[0]->isMemBoundOp()) {
            cur = cur->getPredecessors()[0];
            if (visitTime.count(cur->getGuid()))
                break;
            tmp.emplace_back(cur);
            visitTime.emplace(cur->getGuid(), cnt);
        }
        for (int i = tmp.size() - 1; i >= 0; i--) {
            chainOps.emplace_back(tmp[i]);
        }
        chainOps.emplace_back(op);
        cur = op;
        while (cur->getSuccessors().size() == 1 &&
               cur->getSuccessors()[0]->isMemBoundOp()) {
            cur = cur->getSuccessors()[0];
            if (visitTime.count(cur->getGuid()))
                break;
            chainOps.emplace_back(cur);
            visitTime.emplace(cur->getGuid(), cnt);
        }
        make_ref<GraphObj>(runtimeExec, chainOps)->print();

        auto bestGraph = make_ref<GraphObj>(runtimeExec, chainOps);
        // Eliminate transpose and reshape operators
        if (auto eliminatedGraph = mutator->eliminateVertically(
                make_ref<GraphObj>(runtimeExec, chainOps)))
            bestGraph = eliminatedGraph;
        // Fuse membound operators
        if (auto optGraph = mutator->fuseVertically(bestGraph))
            bestGraph = optGraph;
        for (auto op : bestGraph->getOperators()) {
            ops.emplace_back(op);
        }
    }
    if (ops.empty()) {
        IT_TODO_HALT();
        IT_ASSERT(graph->getOutputs().size() == 1);
        IT_ASSERT(graph->getInputs().size() == 1);
        // auto g = make_ref<GraphObj>(runtime);
        // TODO: add identity
        ops.emplace_back(make_ref<ReshapeObj>(nullptr, graph->getInputs()[0],
                                              graph->getOutputs()[0]));
    }
    return make_ref<GraphObj>(runtimeExec, ops);
}

} // namespace infini
