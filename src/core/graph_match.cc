#include "core/graph_match.h"

namespace infini {
Ref<GraphMatchObj> GraphMatchObj::clone() {
    auto newObj = make_ref<GraphMatchObj>(pattern);
    newObj->ops = ops;
    newObj->opMap = opMap;
    newObj->opMapRevese = opMapRevese;
    newObj->tensorMap = tensorMap;
    return newObj;
}

void GraphMatchObj::addOp(const Operator &anchorOp, const Operator &patternOp) {
    ops.emplace(anchorOp);
    opMap.emplace(anchorOp, patternOp);
    opMapRevese.emplace(patternOp, anchorOp);
    recordOutsideTensorMap(patternOp, anchorOp);
}

TensorVec GraphMatchObj::getInputs() const {
    TensorVec ret;
    for (auto t : pattern->getInputsFromOutside()) {
        IT_ASSERT(tensorMap.count(t) > 0);
        ret.push_back(tensorMap.at(t));
    }
    return ret;
}

TensorVec GraphMatchObj::getOutputs() const {
    TensorVec ret;
    for (auto t : pattern->getOutputs2Outside()) {
        IT_ASSERT(tensorMap.count(t) > 0);
        ret.push_back(tensorMap.at(t));
    }
    return ret;
}

std::string GraphMatchObj::toString() const {
    std::ostringstream oss;
    oss << "MatchGraph operators:\n";
    for (const auto &op : ops) {
        vector<UidBaseType> preds, succs;
        for (auto &o : op->getPredecessors())
            preds.emplace_back(o->getGuid());
        for (auto &o : op->getSuccessors())
            succs.emplace_back(o->getGuid());
        oss << "OP " << op->getGuid();
        oss << ", pred " << vecToString(preds);
        oss << ", succ " << vecToString(succs);
        oss << ", " << op << "\n";
    }
    return oss.str();
}

// if the input pattern tensor is from outside,find the
// corresponding input anchor tensor,and record.
void GraphMatchObj::recordOutsideTensorMap(const Operator &patternOp,
                                           const Operator &anchorOp) {
    for (size_t i = 0; i < patternOp->getInputs().size(); ++i) {
        if (pattern->isInputFromOutside(patternOp->getInputs(i)))
            tensorMap.emplace(patternOp->getInputs(i), anchorOp->getInputs(i));
    }
    for (size_t i = 0; i < patternOp->getOutputs().size(); ++i) {
        if (pattern->isOutput2Outside(patternOp->getOutput(i)))
            tensorMap.emplace(patternOp->getOutput(i), anchorOp->getOutput(i));
    }
}

SubGraphObj::SubGraphObj(Runtime runtime, const TensorVec &inputs)
    : GraphObj(runtime), ins(inputs) {
    for (auto t : ins)
        tensors.push_back(t);
}

vector<MatchGraph> SubGraphRewriter::findMatch(const SubGraph &pattern) {
    this->pattern = pattern;
    vector<MatchGraph> matches;
    bool firstHead = true, retStatus = true;
    for (auto input : pattern->getInputsFromOutside()) {
        auto inputOf = input->getTargets();
        for (auto opHead : inputOf) {
            if (std::find(pattern->getOperators().begin(),
                          pattern->getOperators().end(),
                          opHead) == pattern->getOperators().end())
                continue;                             // not belongs to pattern
            if (opHead->getPredecessors().size() > 0) // not a head
                continue;
            if (firstHead) {
                firstHead = false;
                if (!findMatch(nullptr, nullptr, opHead, matches)) {
                    retStatus = false;
                    break;
                }
            } else {
                if (!findMatch2(nullptr, nullptr, opHead, matches)) {
                    retStatus = false;
                    break;
                }
            }
        }
        if (!retStatus)
            break;
    }

    vector<MatchGraph> ret;
    for (auto match : matches) {
        if (checkMatchValid(match))
            ret.push_back(match);
    }
    return ret;
}

bool SubGraphRewriter::findMatch(const MatchGraph &gLastMatch,
                                 const Operator &opLastMatch,
                                 const Operator &opPattern,
                                 vector<MatchGraph> &gMatch) {
    OpVec candidates =
        opLastMatch ? opLastMatch->getSuccessors() : graph->getOperators();
    OpLists nodesMatch =
        matchInCandidates(candidates, opPattern, pattern->isHead(opPattern),
                          pattern->isTail(opPattern));

    IT_ASSERT(nodesMatch.size() <= 1 || !opLastMatch);
    updateMatchedGraph(gLastMatch, nodesMatch, gMatch, opPattern);

    if (nodesMatch.size() == 0) {
        return false;
    }

    // dst is matched, process successors recursively
    for (auto successorPattern : opPattern->getSuccessors()) {

        bool bRet = false;
        if (opLastMatch) {
            IT_ASSERT(nodesMatch.size() == 1);
            if (gLastMatch->hasMatched(successorPattern))
                continue;
            bRet = findMatch(gLastMatch, nodesMatch.front(), successorPattern,
                             gMatch);
        } else {
            IT_ASSERT(nodesMatch.size() == gMatch.size());
            auto tmp1 = gMatch;
            auto itr1 = nodesMatch.begin();
            auto itr2 = gMatch.begin();
            for (; itr1 != nodesMatch.end() && itr2 != gMatch.end(); ++itr2) {
                if (findMatch(*itr2, *itr1, successorPattern, tmp1)) {
                    bRet = true;
                    ++itr1;
                } else
                    itr1 = nodesMatch.erase(itr1);
            }
            gMatch = tmp1;
        }
        // not found,return false
        if (!bRet) {
            return false;
        }
    }
    return true;
}

bool SubGraphRewriter::findMatch2(const MatchGraph &gLastMatch,
                                  const Operator &opLastMatch,
                                  const Operator &opPattern,
                                  vector<MatchGraph> &matches) {
    vector<MatchGraph> curMatches;
    for (auto match : matches) {
        OpVec candidates =
            opLastMatch ? opLastMatch->getSuccessors() : graph->getOperators();
        // filter candiates in matches
        for (auto itr2 = candidates.begin(); itr2 != candidates.end();) {
            if (match->hasContained(
                    *itr2)) // already belonged to the matched sub graph
                itr2 = candidates.erase(itr2);
            else
                ++itr2;
        }

        OpLists nodesMatch = matchInCandidates(
            candidates, opPattern, opPattern->getPredecessors().size() == 0,
            opPattern->getSuccessors().size() == 0);

        // no match nodes found, do not add the match to curMatches, continue
        if (nodesMatch.size() == 0) {
            continue;
        }

        for (auto node : nodesMatch) {
            auto curMatch = match->clone();
            curMatch->addOp(node, opPattern); // anchor and pattern

            // add to curMatches
            curMatches.push_back(curMatch);

            // dst is matched, process successors recursively
            for (auto successorPattern : opPattern->getSuccessors()) {
                if (match->hasMatched(successorPattern)) // has already matched
                    continue;
                if (!findMatch(curMatch, node, successorPattern, curMatches)) {
                    // curMatch has been removed from curMatches in
                    // "findMatch",so just break
                    break;
                }
            }
        }
    }
    matches = curMatches;
    return true;
}

OpLists SubGraphRewriter::matchInCandidates(const OpVec &ops,
                                            const Operator &opPattern,
                                            bool isHead, bool isTail) {
    OpLists ret;
    for (auto op : ops) {
        if (MatchNode(opPattern, op, isHead, isTail))
            ret.push_back(op);
    }
    return ret;
}

bool SubGraphRewriter::MatchNode(const Operator &a, const Operator &b,
                                 bool isHead, bool isTail) const {
    if (a->getOpType() != b->getOpType())
        return false;
    if (a->hash() != b->hash())
        return false;

    if (!isHead)
        if (a->getPredecessors().size() != b->getPredecessors().size())
            return false;

    if (!isTail)
        if (a->getSuccessors().size() != b->getSuccessors().size())
            return false;
    return true;
};

void SubGraphRewriter::updateMatchedGraph(const MatchGraph &gLastMatch,
                                          OpLists &opMatch,
                                          vector<MatchGraph> &gMatch,
                                          Operator opPattern) {
    if (opMatch.size() == 0) {
        if (nullptr != gLastMatch) {
            auto pos = std::find(gMatch.begin(), gMatch.end(), gLastMatch);
            IT_ASSERT(pos != gMatch.end());
            gMatch.erase(pos);
        }
    } else {
        // anchor is a head
        if (nullptr == gLastMatch) {
            for (auto op : opMatch) {
                auto match = make_ref<GraphMatchObj>(pattern);
                match->addOp(op, opPattern);
                gMatch.push_back(match);
            }
        } else {
            IT_ASSERT(opMatch.size() == 1);
            gLastMatch->addOp(opMatch.front(), opPattern);
        }
    }
}

bool SubGraphRewriter::checkOverlapsWithPreviousMatch(
    const MatchGraph &match,
    const std::unordered_set<Operator> &nodesToDelete) const {
    for (auto op : match->getOps()) {
        if (nodesToDelete.count(op) > 0)
            return false;
    }
    return true;
}

bool SubGraphRewriter::checkMatchValid(const MatchGraph &match) const {
    for (auto t : pattern->getInputsFromOutside()) {
        auto tAnchor = match->getAnchorByPattern(t);
        // the corrresponding precessor must not belong to the match
        auto preOpAnchor = tAnchor->getSource();
        if (preOpAnchor && match->hasContained(preOpAnchor)) {
            return false;
        }
    }
    // check  connections
    for (auto opPattern : pattern->getOperators()) {
        auto opAnchor = match->getAnchorByPattern(opPattern);
        for (auto prePattern : opPattern->getPredecessors()) {
            auto preAnchor = match->getAnchorByPattern(prePattern);
            auto ops = opAnchor->getPredecessors();
            if (std::find(ops.begin(), ops.end(), preAnchor) == ops.end())
                return false;
            ops = preAnchor->getSuccessors();
            if (std::find(ops.begin(), ops.end(), opAnchor) == ops.end())
                return false;
        }
    }
    return true;
}

// replace all sub graphs which matched subA with subB in g
void SubGraphRewriter::replaceSubGraph(const SubGraph &pattern,
                                       const SubGraph &replacement) {
    IT_ASSERT(checkReplacement(pattern, replacement));
    this->pattern = pattern;

    // find matches in graph.
    auto matches = findMatch(pattern);

    std::unordered_set<Operator> nodesToDelete;
    map<Tensor, Tensor> replaceMap;
    map<Tensor, Tensor> replaceMapReverse;
    for (auto match : matches) {
        // matches may overlap with eachother. if some operator has been in
        // another folded match,we must skip this one
        if (!checkOverlapsWithPreviousMatch(match, nodesToDelete))
            continue;

        auto inputs = match->getInputs();
        for (auto &input : inputs) {
            if (replaceMap.count(input) > 0)
                input = replaceMap[input];
        }
        auto outputs = match->getOutputs();

        // first, remove old successors for input
        for (auto input : inputs) {
            for (auto op : input->getTargets()) {
                if (match->hasContained(op)) {
                    graph->deleteConnection(input, op);
                }
            }
        }

        // second, insert replacement sub graph to graph.
        auto newOutputs = addSubGraph(replacement, inputs);

        // check replaced outputs and record
        IT_ASSERT(outputs.size() == newOutputs.size());
        for (size_t i = 0; i < outputs.size(); ++i) {
            IT_ASSERT(isReplacable(outputs[i], newOutputs[i]));
            replaceMap.emplace(outputs[i], newOutputs[i]);
            replaceMapReverse.emplace(newOutputs[i], outputs[i]);
        }

        // third, change connections for new output
        for (auto output : outputs) {
            auto successors = output->getTargets();
            for (auto successor : successors) {
                auto newOutput = replaceMap[output];
                graph->replaceConnection(output, newOutput, successor);
            }
        }

        // record ops need to delete
        for (auto op : match->getOps())
            nodesToDelete.insert(op);

        // remove match from graph
        for (auto op : match->getOps()) {
            for (auto tensor : op->getInputs()) {
                if (replaceMapReverse.count(tensor) > 0)
                    tensor = replaceMapReverse[tensor];
                if (std::find(inputs.begin(), inputs.end(), tensor) ==
                    inputs.end()) {
                    graph->removeTensor(tensor);
                }
            }
            for (auto tensor : op->getOutputs()) {
                graph->removeTensor(tensor);
            }
            graph->removeOperator(op);
        }

        IT_ASSERT(graph->checkValid());
    }
}

// "inputs" must be tensors in original graph
TensorVec SubGraphRewriter::addSubGraph(const SubGraph &g,
                                        const TensorVec &inputs) {
    // check inputs
    for (auto input : inputs) {
        auto tensors = graph->getTensors();
        IT_ASSERT(std::find(tensors.begin(), tensors.end(), input) !=
                  tensors.end());
    }

    // check compatible with sub graph
    auto ins = g->getInputsFromOutside();
    IT_ASSERT(checkReplacement(ins, inputs));

    std::map<Tensor, Tensor> tensorMap;
    for (size_t i = 0; i < ins.size(); ++i) {
        tensorMap.emplace(ins[i], inputs[i]);
    }

    for (auto t : g->getTensors()) {
        if (tensorMap.find(t) == tensorMap.end()) {
            auto tClone = graph->addTensor(t->getDims(), t->getDType());
            tensorMap.emplace(t, tClone);
        }
    }

    for (auto op : g->getOperators()) {
        TensorVec inputs, outputs;
        for (auto t : op->getInputs()) {
            inputs.push_back(tensorMap.at(t));
        }
        for (auto t : op->getOutputs()) {
            outputs.push_back(tensorMap.at(t));
        }
        graph->cloneOperator(op, inputs, outputs);
    }

    TensorVec out;
    for (auto t : g->getOutputs2Outside()) {
        out.push_back(tensorMap[t]);
    }
    return out;
}

void SubGraphRewriter::removeSubGraph(MatchGraph match) {
    TensorVec inputs = match->getInputs();

    for (auto op : match->getOps()) {
        for (auto tensor : op->getInputs()) {
            if (std::find(inputs.begin(), inputs.end(), tensor) ==
                inputs.end()) {
                graph->removeTensor(tensor);
            }
        }

        for (auto tensor : op->getOutputs()) {
            graph->removeTensor(tensor);
        }
        graph->removeOperator(op);
    }
}

// inputs and outputs must be appointed.
bool SubGraphRewriter::checkReplacement(const SubGraph &pattern,
                                        const SubGraph &other) const {
    return checkReplacement(pattern->getInputsFromOutside(),
                            other->getInputsFromOutside()) &&
           checkReplacement(pattern->getOutputs2Outside(),
                            other->getOutputs2Outside()) &&
           pattern->getInputsFromOutside().size() != 0 &&
           pattern->getOutputs2Outside().size() != 0;
}

bool SubGraphRewriter::checkReplacement(const TensorVec &left,
                                        const TensorVec &right) const {
    if (left.size() != right.size())
        return false;
    for (size_t i = 0; i < left.size(); ++i) {
        if (!isReplacable(left[i], right[i]))
            return false;
    }
    return true;
}

bool SubGraphRewriter::isReplacable(const Tensor &l, const Tensor &r) const {
    return (l->getDType() == r->getDType() && l->getDims() == r->getDims());
}

} // namespace infini
