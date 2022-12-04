#include "core/search_engine.h"
#include "core/hash.h"
#include "core/runtime.h"

#include <algorithm>
#include <iostream>
#include <unordered_set>

namespace infini {

void SearchEngine::printMetaGraph(Ref<SearchEngine::MetaGraph> metaGraph) {
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
    std::cout << "[INFO] perf: " << runtimeExec->getPerfTime(graph)
              << std::endl;

    // TODO: partition graph
    std::vector<Graph> partitions = partitionGraph(graph);

    std::cout << "[INFO] Partition num: " << partitions.size() << std::endl;
    std::vector<Operator> ops;
    std::vector<Graph> subGraphCandidates;
    for (size_t pid = 0; pid < partitions.size(); pid++) {
        auto &subGraph = partitions[pid];
        std::cout << "[INFO] Partition: " << pid << std::endl;
        std::vector<Graph> candidates = search(subGraph);
        double bestTime = 1e8;
        double bestId = -1;
        for (size_t i = 0; i < candidates.size(); i++) {
            double time = runtimeExec->getPerfTime(candidates[i]);
            if (time < bestTime) {
                bestTime = time;
                bestId = i;
            }
        }
        IT_ASSERT(bestId != -1);
        for (auto op : candidates[bestId]->getOperators()) {
            ops.emplace_back(op);
        }
    }

    auto bestGraph = make_ref<GraphObj>(runtimeExec, ops);
    std::cout << "[INFO] unfused graph: " << std::endl;
    std::cout << bestGraph->toString();
    std::cout << "[INFO] perf: " << runtimeExec->getPerfTime(bestGraph)
              << std::endl;

    // bestGraph = fuse(bestGraph);
    // double time = CpuRuntimeObj::getInstance()->getPerfTime(bestGraph);
    // std::cout << "[INFO] fused graph: " << std::endl;
    // std::cout << bestGraph->toString();
    // std::cout << "[INFO] perf: "
    //           << CpuRuntimeObj::getInstance()->getPerfTime(bestGraph)
    //           << std::endl;

    return bestGraph;
}

std::vector<Graph> SearchEngine::search(const Graph &graph) {
    auto metaGraph = buildMetaGraphWithGraph(graph);
    auto mergedGraphs = searchMerge(metaGraph);
    std::cout << "[INFO] merged graphs: " << mergedGraphs.size() << std::endl;

    std::vector<Graph> mutatedGraphs;
    std::vector<Graph> results;
    for (auto mergedGraph : mergedGraphs) {
        auto mutatedGraphs = searchMutation(mergedGraph);
        for (size_t i = 0; i < std::min(mutatedGraphs.size(), GRAPH_SIZE);
             i++) {
            results.emplace_back(mutatedGraphs[i]);
        }
    }

    sort(results.begin(), results.end()); // compare with perf time
    if (results.size() > GRAPH_SIZE) {
        results.resize(GRAPH_SIZE);
    }
    return results;
}

std::shared_ptr<SearchEngine::MetaGraph>
SearchEngine::buildMetaGraphWithGraph(const Graph graph) {
    auto metaGraph = std::make_shared<MetaGraph>();

    int numOps = graph->getOperators().size();
    std::vector<int> cnt(numOps, 0);
    std::unordered_map<int, int> opMap;
    metaGraph->nodes.clear();
    std::vector<int> q(0);
    for (size_t i = 0; i < graph->getOperators().size(); i++) {
        auto &op = graph->getOperators()[i];
        MetaGraph::Node node;
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

std::shared_ptr<SearchEngine::MetaGraph> SearchEngine::buildMetaGraphWithPlan(
    const std::shared_ptr<SearchEngine::MetaGraph> metaGraph,
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

    auto resultMetaGraph = make_ref<MetaGraph>();
    for (auto &group : groups) {
        std::vector<Operator> ops;
        std::unordered_set<int> preSet, sucSet;
        for (auto id : group) {
            MetaGraph::Node node;
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
                IT_ASSERT(sucSet.find(plan[pre]) != sucSet.end());
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

std::vector<std::shared_ptr<SearchEngine::MetaGraph>>
SearchEngine::searchMerge(std::shared_ptr<SearchEngine::MetaGraph> &metaGraph) {
    IT_ASSERT(metaGraph != nullptr);
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

    std::vector<std::shared_ptr<SearchEngine::MetaGraph>> metaGraphs;
    for (auto &curPlan : plans) {
        metaGraphs.emplace_back(buildMetaGraphWithPlan(metaGraph, curPlan));
    }
    return metaGraphs;
}

void SearchEngine::searchMergeDfs(std::shared_ptr<MetaGraph> &metaGraph,
                                  std::vector<int> &plan,
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
        if (ops.size() == 1 || isMergeable(graph)) {
            searchMergeDfs(metaGraph, plan, nextFrontier, plans, planSet);
        }
        plan = planBackup;
        metaGraph = metaGraphBackup;
    }
    return;
}

std::vector<Graph> SearchEngine::searchMutation(
    const std::shared_ptr<SearchEngine::MetaGraph> &metaGraph) {
    std::vector<Graph> graphs = {nullptr};
    // Append a node to all existing candidates
    for (auto &node : metaGraph->nodes) {
        std::vector<Graph> nextGraphs;
        if (node.type == 1) { // If it has computing OPs
            std::vector<Graph> mutatedGraphs = {node.graph};
            // auto mutatedGraphs = mutator->run(node.graph);
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
        std::sort(nextGraphs.begin(), nextGraphs.end(), [&](Graph x, Graph y) {
            return runtimeExec->getPerfTime(x) < runtimeExec->getPerfTime(y);
        });
        if (nextGraphs.size() > GRAPH_SIZE) {
            nextGraphs.resize(GRAPH_SIZE);
        }
        graphs = nextGraphs;
    }
    return graphs;
}

bool SearchEngine::isMergeable(const Graph graph) {
    IT_ASSERT(graph->getOperators().size() <= 1);
    // TODO: mutator is mergeable()
    // auto stat = mutationEngine->statGraph(graph.get());
    // if (stat == Mutator::GroupConv || stat == Mutator::TransposeGroupConv ||
    //     stat == Mutator::BatchMatmul || stat == Mutator::HetConv) {
    //     return 1;
    // }
    return 0;
}

std::vector<Graph> SearchEngine::partitionGraph(const Graph graph) {
    std::vector<Graph> partitions;
    partitions.emplace_back(graph);
    return partitions;

    // reversed DFS post-order is topo-order
    // std::unordered_map<const Operator *, int> preOrder, postOrder;
    // std::vector<Operator *> ops;
    // int preCnt = 0, postCnt = 0;
    // std::function<void(Operator *)> dfs = [&](Operator *op) {
    //     if (preOrder.count(op)) {
    //         return;
    //     }
    //     preOrder[op] = preCnt++;
    //     for (auto &&next : op->getSuccessors()) {
    //         dfs(next);
    //     }
    //     postOrder[op] = postCnt++;
    //     ops.emplace_back(op);
    // };
    // for (auto &&op : graph->getOperators()) {
    //     dfs(op);
    // }

    // std::vector<std::shared_ptr<SubGraph>> ret;
    // std::vector<Operator *> headOps;
    // for (auto i = ops.rbegin(); i != ops.rend(); i++) {
    //     headOps.emplace_back(*i);
    //     if ((*i)->getPredecessors().size() + (*i)->getSuccessors().size() >=
    //             (size_t)partitionThreshold &&
    //         !(*i)->isComputeOp()) {
    //         auto preOrderI = preOrder.at(*i);
    //         auto postOrderI = postOrder.at(*i);
    //         for (auto j = ops.rbegin(); j != i; j++) {
    //             // True predecessor
    //             if (preOrder.at(*j) < preOrderI) {
    //                 for (auto &&k : (*j)->getSuccessors()) {
    //                     if (postOrder.at(k) < postOrderI) {
    //                         goto fail;
    //                     }
    //                 }
    //             }
    //         }
    //         std::shared_ptr<SubGraph> gRest, gPart;
    //         graph->split(gRest, gPart, headOps);
    //         headOps.clear();
    //         ret.emplace_back(std::move(gPart));
    //     }
    // fail:;
    // }
    // if (!headOps.empty()) {
    //     std::shared_ptr<SubGraph> gRest, gPart;
    //     graph->split(gRest, gPart, headOps);
    //     ret.emplace_back(std::move(gPart));
    // }
    // return ret;
}

} // namespace infini