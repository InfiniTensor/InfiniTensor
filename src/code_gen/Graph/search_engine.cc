#include "code_gen/search_engine.h"
#include "code_gen/cmutator.h"
#include "code_gen/nnet/Visitor/FullPrinterVisitor.h"
#include "code_gen/nnet/Visitor/MergeMemboundMutator.h"
#include "code_gen/nnet/dmutator.h"
#include "code_gen/nnet/expr.h"
#include "code_gen/nnet/nmutator.h"
#include "code_gen/perf_engine.h"
#include <algorithm>
#include <iostream>
#include <unordered_set>

namespace tpm {

SearchEngine::SearchEngine(const std::shared_ptr<Mutator> &mutationEngine)
    : mutationEngine(mutationEngine) {
    perfEngine = std::make_shared<PerfEngine>();
    // eliminateEngine = std::make_shared<TransEliminator>();
    auto msenv = getenv("PET_MUTATION_SIZE");
    if (msenv != nullptr)
        MUTATION_SIZE = atoi(msenv);
    auto mdenv = getenv("PET_MUTATION_DEPTH");
    if (mdenv != nullptr)
        MUTATION_DEPTH = atoi(mdenv);
}

SearchEngine::~SearchEngine() {}

bool SearchEngine::Candidate::cmp(const Candidate &a, const Candidate &b) {
    return a.perf < b.perf;
};

int SearchEngine::MetaGraph::print() {
    for (size_t i = 0; i < nodes.size(); i++) {
        auto &node = nodes[i];
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
    return 0;
}

int SearchEngine::run(const std::shared_ptr<SubGraph> &graph,
                      std::shared_ptr<SubGraph> &bestGraph) {
    int err = 0;
    double to = 0, t = 0;
    to = getPerf(graph, true);
    std::cout << "Origin Perf: " << to << std::endl;
    graph->printBrief();
    // Partition
    std::vector<std::shared_ptr<SubGraph>> parts;
    parts = partition(graph);
    std::cout << "Partition size: " << parts.size() << std::endl;
    std::vector<Operator *> ops;
    std::vector<std::shared_ptr<SubGraph>> bestParts;
    int pid = 0;
    for (auto &p : parts) {
        std::cout << "Partition: " << pid << std::endl;
        std::vector<std::shared_ptr<SubGraph>> res;
        err = search(p, res);
        if (err) {
            return 1;
        }
        std::vector<Candidate> candidates(0);
        for (auto g : res) {
            auto tmpGraph = g;
            auto tmpPerf = getPerf(tmpGraph, true);
            candidates.emplace_back(Candidate(tmpGraph, tmpPerf));
        }
        std::sort(candidates.begin(), candidates.end(), Candidate::cmp);
        bestParts.emplace_back(candidates[0].graph);
        pid++;
    }
    for (auto p : bestParts) {
        for (auto op : p->getOperators()) {
            ops.emplace_back(op);
        }
    }
    bestGraph = std::make_shared<SubGraph>(ops);
    t = getPerf(bestGraph, true);
    std::cout << "Best Unfused Perf: " << t << std::endl;
    for (auto op : bestGraph->getOperators()) {
        if (auto memboundOp = dynamic_cast<MemBoundOp *>(op)) {
            if (memboundOp->getExpr())
                dbg(nnet::FullPrinterVisitor().print(memboundOp->getExpr()));
            else
                dbg("memboundOp NO source expr.");
        }
    }
    bestGraph->print();

    if (typeid(*mutationEngine) == typeid(Generator) ||
        typeid(*mutationEngine) == typeid(CMutator))
        bestGraph = fuse(bestGraph);
    else if (typeid(*mutationEngine) == typeid(NMutator))
        bestGraph = strip(bestGraph);

    bestGraph->print();
    t = getPerf(bestGraph, true, false);
    std::cout << "Origin Perf: " << to << std::endl;
    std::cout << "Best Perf without correction: " << t << std::endl;
    t = getPerf(bestGraph, false, true);
    std::cout << "Best Perf with correction: " << t << std::endl;
    t = getTransPerf(bestGraph);
    std::cout << "Transpose perf: " << t << std::endl;
    return 0;
}

int SearchEngine::search(const std::shared_ptr<SubGraph> &graph,
                         std::vector<std::shared_ptr<SubGraph>> &bestGraphs) {
    int err;
    std::shared_ptr<MetaGraph> metaGraph;
    err = split(graph, metaGraph);
    if (err) {
        return 1;
    }
    std::vector<std::shared_ptr<MetaGraph>> metaGraphs;
    err = searchDfs(metaGraph, metaGraphs);
    if (err) {
        return 1;
    }
    std::cout << metaGraphs.size() << std::endl;
    std::vector<Candidate> candidates;
    std::vector<Candidate> result;
    for (auto meta : metaGraphs) {
        err = searchBfs(meta, candidates);
        if (err) {
            return 1;
        }
        for (auto &candidate : candidates) {
            result.emplace_back(candidate);
        }
    }
    sort(result.begin(), result.end(), Candidate::cmp);
    bestGraphs.clear();
    for (int i = 0; i < int(result.size()) && i < GRAPH_SIZE; i++) {
        bestGraphs.emplace_back(result[i].graph);
    }
    return 0;
}

int SearchEngine::split(const std::shared_ptr<SubGraph> &graph,
                        std::shared_ptr<MetaGraph> &metaGraph) {
    int n = graph->getOperators().size();
    auto &opList = graph->getOperators();
    std::vector<int> cnt(n, 0);
    std::unordered_map<int, int> opMap;
    metaGraph = std::make_shared<MetaGraph>();
    metaGraph->nodes.clear();
    std::vector<int> q(0);
    for (int i = 0; i < n; i++) {
        typename MetaGraph::Node node;
        std::vector<Operator *> ops(0);
        ops.emplace_back(opList[i]);
        node.graph = std::make_shared<SubGraph>(ops);
        node.type = opList[i]->isComputeOp();
        node.cnt = opList[i]->getPredecessors().size();
        opMap.emplace(opList[i]->getGuid(), i);
        metaGraph->nodes.emplace_back(node);
    }
    for (int i = 0; i < n; i++) {
        auto &op = opList[i];
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
    return 0;
}

int SearchEngine::searchDfs(
    const std::shared_ptr<MetaGraph> &metaGraph,
    std::vector<std::shared_ptr<MetaGraph>> &metaGraphs) {
    int err = 0;
    metaGraphs.clear();
    int n = metaGraph->nodes.size();
    std::vector<int> frontier(0);
    std::vector<int> f(n);
    for (int i = 0; i < n; i++) {
        f[i] = i;
    }
    for (int i = 0; i < n; i++) {
        if (metaGraph->nodes[i].cnt == 0) {
            frontier.emplace_back(i);
        }
    }
    std::vector<std::vector<int>> candidates(0);
    std::unordered_set<uint64_t> candidateSet;
    candidateSet.clear();
    err = searchDfs(metaGraph, frontier, f, candidates, candidateSet);
    if (err) {
        return 1;
    }
    metaGraphs.clear();
    for (auto &candidate : candidates) {
        std::vector<std::vector<int>> tmp(n, std::vector<int>(0));
        for (int i = 0; i < n; i++) {
            tmp[candidate[i]].emplace_back(i);
        }
        auto meta = std::make_shared<MetaGraph>();
        for (int i = 0; i < n; i++) {
            if (tmp[i].size() == 0) {
                continue;
            }
            std::unordered_set<int> set;
            std::vector<Operator *> ops;
            typename MetaGraph::Node node;
            for (auto id : tmp[i]) {
                for (auto op : metaGraph->nodes[id].graph->getOperators()) {
                    ops.emplace_back(op);
                }
                for (size_t j = 0; j < metaGraph->nodes[id].suc.size(); j++) {
                    int suc = candidate[metaGraph->nodes[id].suc[j]];
                    if (set.find(suc) == set.end()) {
                        node.suc.emplace_back(suc);
                        set.emplace(suc);
                    }
                }
                for (size_t j = 0; j < metaGraph->nodes[id].pre.size(); j++) {
                    int pre = candidate[metaGraph->nodes[id].pre[j]];
                    if (set.find(pre) == set.end()) {
                        node.pre.emplace_back(pre);
                        set.emplace(pre);
                    }
                }
            }
            node.graph = std::make_shared<SubGraph>(ops);
            node.cnt = node.pre.size();
            node.type = ops[0]->isComputeOp();
            meta->nodes.emplace_back(node);
        }
        metaGraphs.emplace_back(meta);
    }

    return 0;
}

int SearchEngine::searchDfs(const std::shared_ptr<MetaGraph> &metaGraph,
                            std::vector<int> &frontier, std::vector<int> &f,
                            std::vector<std::vector<int>> &candidates,
                            std::unordered_set<uint64_t> &candidateSet) {
    int err;
    int n = f.size(), m = frontier.size();
    if (m == 0) {
        std::unordered_map<int, int> map;
        map.clear();
        int cnt = 0;
        for (int i = 0; i < n; i++) {
            if (map.find(f[i]) == map.end()) {
                map.emplace(f[i], cnt);
                cnt++;
            }
            f[i] = map[f[i]];
        }
        uint64_t hash = 0;
        for (int i = 0; i < n; i++) {
            hash = hashAppend(hash, f[i]);
        }
        if (candidateSet.find(hash) != candidateSet.end()) {
            return 0;
        }
        candidateSet.emplace(hash);
        int t = candidates.size();
        candidates.emplace_back(0);
        for (int i = 0; i < n; i++) {
            candidates[t].emplace_back(f[i]);
        }
        return 0;
    }

    std::vector<int> nextFrontier;
    nextFrontier.clear();
    int nn = 0;
    for (auto x : frontier) {
        if (metaGraph->nodes[x].type == 0) {
            nn++;
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
    if (nn > 0) {
        err = searchDfs(metaGraph, nextFrontier, f, candidates, candidateSet);
        if (err) {
            return 1;
        }
        for (auto x : frontier) {
            if (metaGraph->nodes[x].type == 0) {
                for (auto y : metaGraph->nodes[x].suc) {
                    metaGraph->nodes[y].cnt++;
                }
            }
        }
        return 0;
    }

    std::vector<Operator *> ops;
    std::shared_ptr<SubGraph> g;
    std::vector<int> fc(n);
    for (int i = 0; i < n; i++) {
        fc[i] = f[i];
    }

    for (int mask = (1 << m) - 1; mask > 0; mask--) {
        int fa = -1;
        nextFrontier.clear();
        ops.clear();
        for (int i = 0; i < m; i++) {
            auto x = frontier[i];
            if ((1 << i) & mask) {
                if (fa == -1) {
                    fa = f[x];
                } else {
                    f[x] = fa;
                }
                for (auto y : metaGraph->nodes[x].suc) {
                    metaGraph->nodes[y].cnt--;
                    if (metaGraph->nodes[y].cnt == 0) {
                        nextFrontier.emplace_back(y);
                    }
                }
                for (auto op : metaGraph->nodes[x].graph->getOperators()) {
                    ops.emplace_back(op);
                }
            } else {
                nextFrontier.emplace_back(x);
            }
        }
        g = std::make_shared<SubGraph>(ops);
        if (isMergeable(g)) {
            err =
                searchDfs(metaGraph, nextFrontier, f, candidates, candidateSet);
            if (err) {
                return 1;
            }
        }
        for (int i = 0; i < m; i++) {
            auto x = frontier[i];
            if ((1 << i) & mask) {
                for (auto y : metaGraph->nodes[x].suc) {
                    metaGraph->nodes[y].cnt++;
                }
            }
        }
        for (int i = 0; i < n; i++) {
            f[i] = fc[i];
        }
    }
    return 0;
}

int SearchEngine::searchBfs(const std::shared_ptr<MetaGraph> &metaGraph,
                            std::vector<Candidate> &candidates) {
    int err = 0;
    std::cout << "start search bfs." << std::endl;
    candidates.clear();
    std::vector<Operator *> ops;
    candidates.emplace_back(Candidate(nullptr, 0));
    for (auto &node : metaGraph->nodes) {
        std::vector<Candidate> tmp(0);
        std::vector<std::shared_ptr<SubGraph>> mutatedGraphs;
        if (node.type == 1) {
            err = getMutation(node.graph, mutatedGraphs);
            if (err) {
                return 1;
            }
            for (auto candidate : candidates) {
                for (auto &g : mutatedGraphs) {
                    std::vector<Operator *> ops;
                    if (candidate.graph != nullptr) {
                        for (auto op : candidate.graph->getOperators()) {
                            ops.emplace_back(op);
                        }
                    }
                    for (auto op : g->getOperators()) {
                        ops.emplace_back(op);
                    }
                    auto tmpGraph = std::make_shared<SubGraph>(ops);
                    auto tmpPerf = getPerf(tmpGraph);
                    tmp.emplace_back(Candidate(tmpGraph, tmpPerf));
                }
            }
        } else {
            for (auto candidate : candidates) {
                std::vector<Operator *> ops;
                if (candidate.graph != nullptr) {
                    for (auto op : candidate.graph->getOperators()) {
                        ops.emplace_back(op);
                    }
                }
                for (auto op : node.graph->getOperators()) {
                    ops.emplace_back(op);
                }
                auto tmpGraph = std::make_shared<SubGraph>(ops);
                auto tmpPerf = getPerf(tmpGraph);
                tmp.emplace_back(Candidate(tmpGraph, tmpPerf));
            }
        }
        std::sort(tmp.begin(), tmp.end(), Candidate::cmp);
        candidates.clear();
        for (int i = 0; i < int(tmp.size()) && i < GRAPH_SIZE; i++) {
            candidates.emplace_back(tmp[i]);
        }
    }
    std::cout << "end search bfs." << std::endl;
    return 0;
}

int SearchEngine::isMergeable(const std::shared_ptr<SubGraph> &graph) {
    if (graph->getOperators().size() <= 1) {
        return 1;
    }
    auto stat = mutationEngine->statGraph(graph.get());
    if (stat == Mutator::GroupConv || stat == Mutator::TransposeGroupConv ||
        stat == Mutator::BatchMatmul || stat == Mutator::HetConv) {
        return 1;
    }
    return 0;
}

int SearchEngine::isMutatable(const std::shared_ptr<SubGraph> &graph) {
    std::cout << "[ERROR] search_engine::isMutatable: function not impl."
              << std::endl;
    return 0;
}

int SearchEngine::isSpecialMutation(Operator *op, int depth) {
    std::vector<Operator *> ops;
    ops.emplace_back(op);
    auto graph = std::make_shared<SubGraph>(ops);
    auto stat = mutationEngine->statGraph(graph.get());
    if (stat == Mutator::NormalOddConv && (depth < 3)) {
        return 1;
    }
    return 0;
}

double SearchEngine::getPerf(const std::shared_ptr<SubGraph> &graph,
                             bool profiling, bool withPenalty) {
    double totTime = 0;
    std::map<Operator::OpType, double> opTime; // Statistics
    std::map<Operator::OpType, int> opCnt;
    perfEngine->setPenalty(withPenalty);
    if (profiling)
        puts("\n========== PET graph getPerf ============");
    for (auto op : graph->getOperators()) {
        double t = op->perf(perfEngine.get(), 200, 200);
        if (profiling) {
            op->print();
            printf(" op_time %lf\n", t);
            opTime[op->getType()] += t;
            opCnt[op->getType()]++;
        }
        totTime += t;
    }
    if (profiling) {
        printf("%11s %3s %7s %7s %7s\n", "Op", "Cnt", "T_tot", "Percent",
               "T_mean");
        for (const auto &[type, t] : opTime) {
            printf("%11s %3d %7.3f %7.1f %7.3f\n",
                   Operator::getOpName(type).data(), opCnt[type], t,
                   t / totTime * 100, t / opCnt[type]);
        }
    }
    return totTime;
}

double SearchEngine::getMaxPerf(const std::shared_ptr<SubGraph> &graph,
                                bool profiling, bool withPenalty) {
    double time = 0;
    perfEngine->setPenalty(withPenalty);
    for (auto op : graph->getOperators()) {
        double t = op->perf(perfEngine.get(), 200, 200);
        time = std::max(time, t);
    }
    return time;
}

double SearchEngine::getTransPerf(const std::shared_ptr<SubGraph> &graph) {
    double time = 0;
    for (auto op : graph->getOperators()) {
        if (op->isTransposeOp()) {
            double t = op->perf(perfEngine.get(), 200, 200);
            time += t;
        }
        // print detailed perf data
        // auto t = op->perf(perfEngine.get(), 10);
        // time += t;
        // printf("%s %f\n", op->toString().data(), t);
    }
    return time;
}

// get mutations after MUTATION_DEPTH rounds.
int SearchEngine::getMutation(
    std::shared_ptr<SubGraph> &graph,
    std::vector<std::shared_ptr<SubGraph>> &mutatedGraphs) {
    // return archived mutation if existed.
    uint64_t graphHash = graph->getHash();
    if (mutationArchive.find(graphHash) != mutationArchive.end()) {
        auto &archive = mutationArchive[graphHash];
        mutatedGraphs.clear();
        for (auto g : archive) {
            mutatedGraphs.emplace_back(g);
        }
        return 0;
    }

    std::cout << "get Mutation: " << graphHash << std::endl;
    std::vector<Operator *> corpOps;
    std::vector<Operator *> restOps;
    for (auto op : graph->getOperators()) {
        if (op->isComputeOp()) {
            corpOps.emplace_back(op);
        } else {
            restOps.emplace_back(op);
        }
    }

    if (restOps.size() > 0) {
        std::cout << "[ERROR] search_engine::getMutation: input graph has "
                     "non-compute op."
                  << std::endl;
        return 1;
    }

    std::shared_ptr<SubGraph> corp;
    std::vector<std::shared_ptr<SubGraph>> baseGraphs;
    std::vector<SubGraph *> mutation;
    if (corpOps.size() == 0) {
        std::cout << "[ERROR] search_engine::getMutation: no compute op, "
                     "invalid candidate.type."
                  << std::endl;
        return 1;
    }
    if (corpOps.size() == 1) {
        baseGraphs.emplace_back(graph);
    }
    if (corpOps.size() > 1) {
        corp = std::make_shared<SubGraph>(corpOps);
        if (!isMergeable(corp)) {
            std::cout
                << "[ERROR] search_engine::getMutation: ops can't be merge."
                << std::endl;
            return 1;
        }
        mutationEngine->run(corp.get(), mutation);
        if (mutation.size() == 0) {
            std::cout << "[WARNING] search_engine::getMutation: mergeable "
                         "subgraph can't be merged. (mutation engine bug. "
                         "should be ERROR)"
                      << std::endl;
            corp->print();
            // return 1;
        }
        for (auto tmpGraph : mutation) {
            baseGraphs.emplace_back(
                std::make_shared<SubGraph>(tmpGraph->getOperators()));
        }
    }

    std::unordered_set<uint64_t> mutationSet;
    std::vector<Candidate> q;
    std::vector<int> f;
    Operator *computeOp;
    uint64_t mutationHash;
    for (auto baseGraph : baseGraphs) {
        corpOps.clear();
        restOps.clear();
        for (auto op : baseGraph->getOperators()) {
            if (op->isComputeOp()) {
                corpOps.emplace_back(op);
            } else {
                restOps.emplace_back(op);
            }
        }
        if (corpOps.size() == 0) {
            std::cout << "[ERROR] search_engine::getMutation: baseGraph have "
                         "no compute ops."
                      << std::endl;
            return 1;
        }
        if (corpOps.size() > 1) {
            std::cout << "[ERROR] search_engine::getMutation: baseGraph have "
                         "multiple compute ops."
                      << std::endl;
            return 1;
        }
        computeOp = corpOps[0];

        mutationHash = getMutationHash(computeOp);
        mutationSet.emplace(mutationHash);
        q.emplace_back(baseGraph, getPerf(baseGraph));
        f.emplace_back(0);
    }

    for (size_t i = 0;; i++) {
        if (i >= q.size()) {
            break;
        }
        if (f[i] >= MUTATION_DEPTH) {
            continue;
        }

        corpOps.clear();
        restOps.clear();
        for (auto op : q[i].graph->getOperators()) {
            if (op->isComputeOp()) {
                corpOps.emplace_back(op);
            } else {
                restOps.emplace_back(op);
            }
        }
        if (corpOps.size() == 0) {
            std::cout << "[ERROR] search_engine::getMutation: search graph "
                         "have no compute ops."
                      << std::endl;
            return 1;
        }
        if (corpOps.size() > 1) {
            std::cout << "[ERROR] search_engine::getMutation: search graph "
                         "have multiple compute ops."
                      << std::endl;
            return 1;
        }
        corp = std::make_shared<SubGraph>(corpOps);
        mutation.clear();
        mutationEngine->run(corp.get(), mutation);

        for (auto tmpGraph : mutation) {
            corpOps.clear();
            for (auto op : tmpGraph->getOperators()) {
                if (op->isComputeOp()) {
                    corpOps.emplace_back(op);
                }
            }
            if (corpOps.size() == 0) {
                std::cout
                    << "[ERROR] search_engine::getMutation: mutation graph "
                       "have no compute ops."
                    << std::endl;
                return 1;
            }

            int nextDepth;
            if (corpOps.size() == 1) {
                computeOp = corpOps[0];
                if (computeOp->getType() == Operator::Conv ||
                    computeOp->getType() == Operator::G2BMM ||
                    computeOp->getType() == Operator::GBMML) {
                    mutationHash = getMutationHash(computeOp);
                    if (mutationSet.find(mutationHash) != mutationSet.end()) {
                        continue;
                    }
                    mutationSet.emplace(mutationHash);
                    // Hack: Special mutation, such as 5 depth mutation.
                    if (isSpecialMutation(computeOp, f[i])) {
                        nextDepth = f[i];
                    } else {
                        nextDepth = f[i] + 1;
                    }
                } else if (computeOp->getType() == Operator::Matmul) {
                    nextDepth = MUTATION_DEPTH;
                } else {
                    // ComputeOp not impl
                    assert(false);
                }
            }
            if (corpOps.size() > 1) {
                nextDepth = MUTATION_DEPTH;
            }

            corpOps.clear();
            for (auto op : tmpGraph->getOperators()) {
                corpOps.emplace_back(op);
            }
            for (auto op : restOps) {
                corpOps.emplace_back(op);
            }
            auto candidateGraph = std::make_shared<SubGraph>(corpOps);
            auto candidatePerf = getPerf(candidateGraph);
            q.emplace_back(candidateGraph, candidatePerf);
            f.emplace_back(nextDepth);
        }
    }

    // select best MUTATION_SIZE graphs.
    std::sort(q.begin(), q.end(), Candidate::cmp);
    mutatedGraphs.clear();
    for (int i = 0; i < int(q.size()) && i < MUTATION_SIZE; i++) {
        mutatedGraphs.emplace_back(q[i].graph);
    }

    // save mutation
    mutationArchive.emplace(graphHash,
                            std::vector<std::shared_ptr<SubGraph>>(0));
    auto &archive = mutationArchive[graphHash];
    for (auto &g : mutatedGraphs) {
        archive.emplace_back(g);
    }

    return 0;
}

// get mutation of a subgraph.
int SearchEngine::getSingleMutation(
    std::shared_ptr<SubGraph> &graph,
    std::vector<std::shared_ptr<SubGraph>> &candidates) {
    int err = 0;
    std::vector<Operator *> computeOps;
    err = graph->getComputeOps(computeOps);
    if (err) {
        return 1;
    }

    std::shared_ptr<SubGraph> rest, corp;
    err = graph->split(rest, corp, computeOps);
    if (err) {
        return 1;
    }

    candidates.clear();
    std::vector<SubGraph *> tmp;
    mutationEngine->run(corp.get(), tmp);
    for (auto g : tmp) {
        g->reset(corp->getInputs(), corp->getOutputs());
        std::shared_ptr<SubGraph> merged;
        std::shared_ptr<SubGraph> frag(g);
        rest->merge(merged, frag);
        candidates.emplace_back(merged);
    }
    return 0;
}

uint64_t SearchEngine::getMutationHash(const Operator *op) {
    uint64_t hash;
    switch (op->getType()) {
    case Operator::Conv:
    case Operator::ConvTrans:
    case Operator::Matmul:
    case Operator::G2BMM:
    case Operator::GBMML:
        hash = mutationEngine->computeHashForSingleComputeOp(op);
        break;
    default:
        std::cout << "[ERROR] search_engine::getMutationHash: invalid input op."
                  << std::endl;
        hash = -1;
    }
    return hash;
}

std::vector<std::shared_ptr<SubGraph>>
SearchEngine::partition(const std::shared_ptr<SubGraph> &graph) {
    // reversed DFS post-order is topo-order
    std::unordered_map<const Operator *, int> preOrder, postOrder;
    std::vector<Operator *> ops;
    int preCnt = 0, postCnt = 0;
    std::function<void(Operator *)> dfs = [&](Operator *op) {
        if (preOrder.count(op)) {
            return;
        }
        preOrder[op] = preCnt++;
        for (auto &&next : op->getSuccessors()) {
            dfs(next);
        }
        postOrder[op] = postCnt++;
        ops.emplace_back(op);
    };
    for (auto &&op : graph->getOperators()) {
        dfs(op);
    }

    std::vector<std::shared_ptr<SubGraph>> ret;
    std::vector<Operator *> headOps;
    for (auto i = ops.rbegin(); i != ops.rend(); i++) {
        headOps.emplace_back(*i);
        if ((*i)->getPredecessors().size() + (*i)->getSuccessors().size() >=
                (size_t)partitionThreshold &&
            !(*i)->isComputeOp()) {
            auto preOrderI = preOrder.at(*i);
            auto postOrderI = postOrder.at(*i);
            for (auto j = ops.rbegin(); j != i; j++) {
                // True predecessor
                if (preOrder.at(*j) < preOrderI) {
                    for (auto &&k : (*j)->getSuccessors()) {
                        if (postOrder.at(k) < postOrderI) {
                            goto fail;
                        }
                    }
                }
            }
            std::shared_ptr<SubGraph> gRest, gPart;
            graph->split(gRest, gPart, headOps);
            headOps.clear();
            ret.emplace_back(std::move(gPart));
        }
    fail:;
    }
    if (!headOps.empty()) {
        std::shared_ptr<SubGraph> gRest, gPart;
        graph->split(gRest, gPart, headOps);
        ret.emplace_back(std::move(gPart));
    }
    return ret;
}

std::shared_ptr<SubGraph>
SearchEngine::fuse(const std::shared_ptr<SubGraph> &graph) {
    std::shared_ptr<SubGraph> tmpGraph(new SubGraph(graph->getOperators()));
    for (Operator *op : tmpGraph->getOperators()) {
        if (op->getType() == Operator::Activation) {
            while (op->getPredecessor() != nullptr &&
                   op->getPredecessor()->getType() == Operator::Transpose &&
                   op->getPredecessor()->getSuccessors().size() == 1) {
                auto pred = op->getPredecessor();
                auto a = pred->getInputs()[0];
                auto b = pred->getOutput();
                auto c = op->getOutput();
                b->setDims(a->getDims());
                op->setInputs({a});
                op->setOutputs({b});
                pred->setInputs({b});
                pred->setOutputs({c});
                tmpGraph->cleanConnection();
                tmpGraph->updateConnection();
                b->setPenalty(a->getPenalty());
            }
        }
    }

    std::vector<Operator *> newOps, removedOps;
    for (Operator *op : tmpGraph->getOperators()) {
        if (op->getType() == Operator::Conv ||
            op->getType() == Operator::Matmul) {
            // If there are more than one successors, we should do CSE first
            if (op->getSuccessors().size() == 1) {
                Operator *succ = op->getSuccessors().front();
                if (succ->getType() == Operator::Activation) {
                    Operator *newOp = op->clone();
                    auto actType = ((ActivationOp *)succ)->getActType();
                    if (op->getType() == Operator::Conv)
                        ((ConvOp *)newOp)->setAct(actType);
                    else
                        ((MatmulOp *)newOp)->setAct(actType);
                    newOp->setInputs(op->getInputs());
                    newOp->setOutputs(succ->getOutputs());
                    newOps.emplace_back(newOp);
                    removedOps.emplace_back(succ);
                    continue;
                }
            }
        }
        newOps.emplace_back(op);
    }
    for (auto &&op : removedOps) {
        newOps.resize(std::remove(newOps.begin(), newOps.end(), op) -
                      newOps.begin());
    }
    std::shared_ptr<SubGraph> newGraph(new SubGraph(newOps));

    return newGraph;
}

int SearchEngine::stripDfs(Operator *op, std::unordered_map<int, int> &f,
                           int flag) {
    std::cout << "[DEBUG]" << op->getType() << " " << op->getInputs().size()
              << " " << op->getOutputs().size() << std::endl;
    assert(op->getInputs().size() == 1 && op->getOutputs().size() == 1);
    f.emplace(op->getGuid(), flag);
    if (op->getPredecessors().size() == 1) {
        auto next = op->getPredecessors()[0];
        if (next->isTransposeOp() && f.find(next->getGuid()) == f.end()) {
            stripDfs(next, f, flag);
        }
    }
    if (op->getSuccessors().size() == 1) {
        auto next = op->getSuccessors()[0];
        if (next->isTransposeOp() && f.find(next->getGuid()) == f.end()) {
            stripDfs(next, f, flag);
        }
    }
    return 0;
}

nnet::Expr transposeOpToExpression(TransposeOp *transposeOp) {
    const auto &AT = transposeOp->getInputs()[0];
    const auto &[before, after] = transposeOp->getBeforeAndAfter();
    for (size_t i = 0; i < before.size(); ++i) {
        assert(before[i].isSingle() && before[i].getSingle() == (int)i);
    }
    const auto A = nnet::makeTensor("A", AT->getDims());
    std::vector<nnet::VarRangePair> loopVarsN;
    nnet::VecExpr subscriptN(after.size());
    for (size_t i = 0; i < after.size(); ++i) {
        assert(after[i].isSingle());
        const auto loopVar =
            nnet::make_ref<nnet::VarNode>("i" + std::to_string(i));
        loopVarsN.emplace_back(
            loopVar, std::make_pair(0, A->getShape(after[i].getSingle())));
        subscriptN[after[i].getSingle()] = loopVar;
    }
    return nnet::makeRangeOperator(loopVarsN, {},
                                   nnet::makeSubscript(A, subscriptN));
}

nnet::Expr toExpr(Operator *op) {
    if (op->getType() == Operator::MemBound) {
        auto memboundOp = dynamic_cast<MemBoundOp *>(op);
        return memboundOp->getExpr();
    }
    if (op->getType() == Operator::Activation) {
        int nDim = op->getInputs()[0]->getDims().size();
        std::vector<int> shape = op->getInputs()[0]->getDims();
        std::vector<int> paddings(nDim);
        std::vector<nnet::Expr> vars(nDim);
        std::vector<nnet::VarRangePair> varRangePair(nDim);
        for (int i = 0; i < nDim; i++) {
            auto var = nnet::make_ref<nnet::VarNode>("var" + std::to_string(i));
            paddings[i] = 0;
            vars[i] = var;
            varRangePair[i] = {var, {0, shape[i]}};
        }
        auto tensor = nnet::make_ref<nnet::TensorNode>("T", shape, paddings);
        auto subscript = makeSubscript(tensor, vars);
        if (op->getType() == Operator::Activation) {
            auto activationOp = dynamic_cast<ActivationOp *>(op);
            if (activationOp->getActType() == Operator::Relu) {
                auto act = nnet::make_ref<nnet::FuncNode>(subscript,
                                                          nnet::FuncType::Relu);
                return nnet::makeRangeOperator(varRangePair, {}, act);
            } else if (activationOp->getActType() == Operator::Tanh) {
                auto act = nnet::make_ref<nnet::FuncNode>(subscript,
                                                          nnet::FuncType::Tanh);
                return nnet::makeRangeOperator(varRangePair, {}, act);
            } else
                nnet_unimplemented_halt();
        }

        // if (activationOp->getActType() == Operator::Sigmoid) {
        //     auto sigmoid = std::make_shared<nnet::FuncNode>(
        //         subscript, nnet::FuncType::Sigmoid);
        //     auto range = nnet::makeRangeOperator(varRangePair, sigmoid);
        // }
    }
    if (op->getType() == Operator::Transpose) {
        return transposeOpToExpression((TransposeOp *)op);
    }
    assert(false);
    return nullptr;
}

Operator *SearchEngine::FuseMemBoundChain(std::vector<Operator *> chainOps) {
    std::cout << "[DEBUG] FuseMemBoundChain" << std::endl;
    if (chainOps.size() == 1) {
        return chainOps[0];
    }
    for (auto &op : chainOps) {
        op->print();
        std::cout << std::endl;
    }
    std::cout << "[DEBUG] end" << std::endl;
    std::vector<nnet::Expr> exprs;
    for (const auto &op : chainOps) {
        assert(op->isMemBoundOp());
        exprs.emplace_back(toExpr(op));
    }
    double maxTime = getMaxPerf(std::make_shared<SubGraph>(chainOps));
    // Fuse a MemboundOp chain
    auto expr = nnet::MergeMemboundMutator(exprs).merge(true);

    // FIXME: use real NNet inputs for verification
    printf("Unimplememnted for verification\n");
    auto memBoundOp =
        new MemBoundOp(chainOps.front()->getInputs(),
                       chainOps.back()->getOutputs(), {}, expr, maxTime);
    memBoundOp->print();
    return memBoundOp;
}

std::shared_ptr<SubGraph>
SearchEngine::strip(const std::shared_ptr<SubGraph> &graph) {
    std::unordered_map<int, int> f;
    std::vector<Operator *> ops;
    int cnt = 0;
    for (auto op : graph->getOperators()) {
        if (f.find(op->getGuid()) != f.end()) {
            continue;
        }
        if (!op->isMemBoundOp() || (op->getPredecessors().size() != 1 &&
                                    op->getSuccessors().size() != 1)) {
            f.emplace(op->getGuid(), ++cnt);
            ops.emplace_back(op);
            continue;
        }
        std::vector<Operator *> chainOps;
        f.emplace(op->getGuid(), ++cnt);

        std::vector<Operator *> tmp;
        auto cur = op;
        while (cur->getPredecessors().size() == 1 &&
               cur->getPredecessors()[0]->isMemBoundOp()) {
            cur = cur->getPredecessors()[0];
            tmp.emplace_back(cur);
            f.emplace(cur->getGuid(), cnt);
        }
        for (int i = tmp.size() - 1; i >= 0; i--) {
            chainOps.emplace_back(tmp[i]);
        }
        chainOps.emplace_back(op);
        cur = op;
        while (cur->getSuccessors().size() == 1 &&
               cur->getSuccessors()[0]->isMemBoundOp()) {
            cur = cur->getSuccessors()[0];
            chainOps.emplace_back(cur);
            f.emplace(cur->getGuid(), cnt);
        }
        int len = chainOps.size();
        std::cout << "[DEBUG] before swap: begin" << std::endl;
        std::make_shared<SubGraph>(chainOps)->print();
        std::cout << "[DEBUG] before swap: end" << std::endl;

        for (int i = 1; i < len; i++) {
            if (!chainOps[i]->isElementWiseOp()) {
                continue;
            }
            int j = i;
            while (j > 0 && chainOps[j - 1]->isTransposeOp()) {
                auto a = chainOps[j - 1], b = chainOps[j];
                auto tmp = a->getOutputs();
                a->setOutputs(b->getOutputs());
                b->setInputs(a->getInputs());
                a->setInputs(tmp);
                b->setOutputs(tmp);
                auto tmpa = a->getSuccessors(), tmpb = b->getPredecessors();
                a->setSuccessors(b->getSuccessors());
                b->setPredecessors(a->getPredecessors());
                a->setPredecessors(tmpb);
                b->setSuccessors(tmpa);
                // Re-compute shape
                b->computeShape();
                a->computeShape();
                chainOps[j - 1] = b;
                chainOps[j] = a;
                j--;
            }
        }
        std::cout << "[DEBUG] after swap: begin" << std::endl;
        std::make_shared<SubGraph>(chainOps)->print();
        std::cout << "[DEBUG] after swap: end" << std::endl;

        ops.emplace_back(FuseMemBoundChain(chainOps));
    }

    return std::make_shared<SubGraph>(ops);
}

std::shared_ptr<PerfEngine> SearchEngine::exportPerfEngine() {
    return perfEngine;
}

} // namespace tpm
