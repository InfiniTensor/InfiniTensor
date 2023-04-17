#pragma once

#include "core/graph.h"
namespace infini {
class SubGraphObj : public GraphObj {
    TensorVec ins;  // inputs from outer predecessors, orders are appointed.
    TensorVec outs; // outputs to outer successors, orders are appointed.

  public:
    SubGraphObj(Runtime runtime, const TensorVec &inputs);
    void setOutputs(const TensorVec &tensors) { outs = tensors; }
    TensorVec getInputsFromOutside() const { return ins; }
    TensorVec getOutputs2Outside() const { return outs; }
    bool isInputFromOutside(Tensor t) const {
        return std::find(ins.begin(), ins.end(), t) != ins.end();
    }
    bool isOutput2Outside(Tensor t) const {
        return std::find(outs.begin(), outs.end(), t) != outs.end();
    }
    bool isHead(const Operator &op) const {
        for (auto in : ins) {
            auto ops = in->getTargets();
            if (std::find(ops.begin(), ops.end(), op) != ops.end())
                return true;
        }
        return false;
    };
    bool isTail(const Operator &op) const {
        for (auto out : outs) {
            if (op == out->getSource())
                return true;
        }
        return false;
    }
};
using SubGraph = Ref<SubGraphObj>;

// Describe a match for subgraph replacement.
class GraphMatchObj {
    std::unordered_set<Operator> ops;
    std::unordered_map<Operator, Operator> opMap;       // anchor->pattern
    std::unordered_map<Operator, Operator> opMapRevese; // pattern->anchor
    std::unordered_map<Tensor, Tensor> tensorMap;       // pattern->anchor
    SubGraph pattern;

  public:
    GraphMatchObj(SubGraph pattern) : pattern(pattern) {}
    Ref<GraphMatchObj> clone();
    void addOp(const Operator &anchorOp, const Operator &patternOp);
    bool hasContained(const Operator &op) const { return opMap.count(op) > 0; }
    bool hasMatched(const Operator &op) const {
        return opMapRevese.count(op) > 0;
    }

    Tensor getAnchorByPattern(const Tensor &t) {
        IT_ASSERT(tensorMap.count(t) > 0);
        return tensorMap.at(t);
    }

    Operator getAnchorByPattern(const Operator &op) {
        IT_ASSERT(opMapRevese.count(op) > 0);
        return opMapRevese.at(op);
    }

    TensorVec getInputs() const;
    TensorVec getOutputs() const;
    std::unordered_set<Operator> getOps() const { return ops; }
    std::string toString() const;

  private:
    void recordOutsideTensorMap(const Operator &patternOp,
                                const Operator &anchorOp);
};
using MatchGraph = Ref<GraphMatchObj>;

class SubGraphRewriter {
    SubGraph pattern;
    Graph graph;

  public:
    SubGraphRewriter(Graph g) : graph(g) {}
    vector<MatchGraph> findMatch(const SubGraph &pattern);
    void replaceSubGraph(const SubGraph &pattern, const SubGraph &replacement);
    TensorVec addSubGraph(const SubGraph &pattern, const TensorVec &inputs);

  private:
    void removeSubGraph(MatchGraph match);
    bool MatchNode(const Operator &a, const Operator &b, bool isHead,
                   bool isTail) const;
    OpLists matchInCandidates(const OpVec &ops, const Operator &opDst,
                              bool isHead, bool isTail);
    bool findMatch(const MatchGraph &lastMatched, const Operator &opLastMatched,
                   const Operator &opDst, vector<MatchGraph> &matched);
    bool findMatch2(const MatchGraph &lastMatched,
                    const Operator &opLastMatched, const Operator &opDst,
                    vector<MatchGraph> &matched);
    void updateMatchedGraph(const MatchGraph &lastMatched, OpLists &opMatched,
                            vector<MatchGraph> &gMatched, Operator dst);

    bool checkReplacement(const SubGraph &pattern, const SubGraph &other) const;
    bool checkReplacement(const TensorVec &left, const TensorVec &right) const;
    bool isReplacable(const Tensor &l, const Tensor &r) const;
    bool checkOverlapsWithPreviousMatch(
        const MatchGraph &match,
        const std::unordered_set<Operator> &nodesToDelete) const;
    bool checkMatchValid(const MatchGraph &match) const;
};
}; // namespace infini
