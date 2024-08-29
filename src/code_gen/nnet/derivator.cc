#include "code_gen/nnet/derivator.h"
#include "code_gen/nnet/Pass/MatchComputationKernel.h"
#include "code_gen/nnet/Pass/MatchMemBoundKernel.h"
#include "code_gen/nnet/Pass/Rule1VariableSplit.h"
#include "code_gen/nnet/Pass/Rule2VariableMerging.h"
#include "code_gen/nnet/Pass/Rule3StageSplit.h"
#include "code_gen/nnet/Pass/Rule4StageMerging.h"
#include "code_gen/nnet/Pass/Rule5RangeRelaxation.h"
#include "code_gen/nnet/Pass/Rule6KenerlMatching.h"
#include "code_gen/nnet/Pass/Rule7DLT.h"
#include "code_gen/nnet/Pass/Rule8GuidedDLT.h"
#include "code_gen/nnet/Pass/Rule90TwoStageElementWise.h"
#include "code_gen/nnet/Pass/Rule91MergeStagesWithSum.h"
#include "code_gen/nnet/Pass/Rule9RangeMagnify.h"
#include "code_gen/nnet/Visitor/CheckOOBVisitor.h"
#include "code_gen/nnet/Visitor/CloneMutator.h"
#include "code_gen/nnet/Visitor/CompareMultiFormulasVisitor.h"
#include "code_gen/nnet/Visitor/CountRoutineVisitor.h"
#include "code_gen/nnet/Visitor/FullPrinterVisitor.h"
#include "code_gen/nnet/Visitor/HashVisitor.h"
#include "code_gen/nnet/Visitor/MergeMemboundMutator.h"
#include "code_gen/nnet/Visitor/Serializer.h"
#include "code_gen/nnet/test.h"

namespace nnet {

class SaveStateGuard {
    Derivator &derivator;

  public:
    SaveStateGuard(Derivator &derivator, const Expr &origin,
                   const string &ruleName, const string &ruleMsg = "")
        : derivator(derivator) {
        derivator.pushIntermediateState(origin);
        derivator.pushRuleState(ruleName);
        derivator.pushRuleMsg(ruleMsg);
    }
    ~SaveStateGuard() {
        derivator.popIntermediateState();
        derivator.popRuleState();
        derivator.popRuleMsg();
    }
};

#define SetUpStateGuard()                                                      \
    SaveStateGuard __guard(*this, origin.root, __FUNCTION__)

void Derivator::dfs(Formula &origin, int depth) {
    guidedSearch(origin, depth);

    if (depth >= maxDepth) {
        return;
    }
    Expr *curExpr = &origin.root;
    nnet_assert((*curExpr)->getType() == NodeType::RangeOpNodeType, __LINE__);
    while ((*curExpr)->getType() == NodeType::RangeOpNodeType) {
        auto curRangeOp = as<RangeOpNode>(*curExpr);
        checkOOB(curRangeOp);
        auto summand = curRangeOp->getSummand();
        if (summand->getType() == NodeType::SubscriptNodeType) {
            auto subscriptOp = as<SubscriptNode>(summand);
            if (rule4StageMerging(origin, depth, *curExpr)) {
                return;
            }
            curExpr = subscriptOp->getObjectPtr();
            nnet_assert(*curExpr != nullptr, __LINE__);
            continue;
        }
        if (summand->getType() == NodeType::BinaryOpNodeType) {
            if (cntAppliedRules[1] < 3)
                rule1VariableSplit(origin, depth, *curExpr); // +1/0
            rule2VariableMerging(origin, depth, *curExpr);   // +1
            if (cntAppliedRules[3] < 1)
                rule3StageSplit(origin, depth, *curExpr);  // +1
            rule5RangeRelaxation(origin, depth, *curExpr); // 0
            rule7DLT(origin, depth, *curExpr);
            rule9RangeMagnify(origin, depth, *curExpr);
            return;
        }
        nnet_unimplemented_halt();
    }
    // RangeOp curRangeOp;
    // for (Expr *curExpr = &origin.root;
    //      curExpr && (curRangeOp = as<RangeOpNode>(*curExpr));) {
    //     checkOOB(curRangeOp);
    //     auto subscript = as<SubscriptNode>(curRangeOp->getSummand());
    //     // isSimplyNested: a directly nested stage
    //     bool isSimplyNested = (subscript &&
    //     subscript->isRangeOpSubscripted()); if (rule4StageMerging(origin,
    //     depth, *curExpr))
    //         return;
    //     // For the next nested stage
    //     curExpr = (isSimplyNested) ? subscript->getObjectPtr() : nullptr;
    // }

    // int stage = 0;
    // for (Expr *curExpr = &origin.root;
    //      curExpr && (curRangeOp = as<RangeOpNode>(*curExpr));) {
    //     stage++;
    //     // isSimplyNested: a directly nested stage
    //     auto subscript = as<SubscriptNode>(curRangeOp->getSummand());
    //     bool isSimplyNested = (subscript &&
    //     subscript->isRangeOpSubscripted());

    //     // TODO recover it
    //     // permuteRangeOps(origin, depth, *curExpr);
    //     // extractSubexpression(origin, depth, *curExpr);

    //     rule4StageMerging(origin, depth, *curExpr);

    //     if (!isSimplyNested) {
    //         std::cout << "num stage: " << depth << " " << stage << std::endl;
    //         if (depth < 5) {
    //             rule1VariableSplit(origin, depth, *curExpr);   // +1/0
    //             rule3StageSplit(origin, depth, *curExpr);      // +1
    //             rule2VariableMerging(origin, depth, *curExpr); // +1
    //             rule5RangeRelaxation(origin, depth, *curExpr); // 0
    //             rule9RangeMagnify(origin, depth, *curExpr);
    //         }
    //         if (depth >= 5) {
    //             rule1VariableSplit(origin, depth, *curExpr);   // +1/0
    //             rule3StageSplit(origin, depth, *curExpr);      // +1
    //             rule2VariableMerging(origin, depth, *curExpr); // +1
    //             rule5RangeRelaxation(origin, depth, *curExpr); // 0
    //             rule6KenerlMatching(origin, depth, *curExpr);  // -1
    //             rule7DLT(origin, depth, *curExpr);             // +1
    //             rule8GuidedDLT(origin, depth, *curExpr);       //
    //             rule9RangeMagnify(origin, depth, *curExpr);
    //         }
    //     }
    //     // For the next nested stage
    //     curExpr = (isSimplyNested) ? subscript->getObjectPtr() : nullptr;
    // }
}

Derivator::Derivator(int maxDepth, bool enableHashPruning, LogMode logMode,
                     PassMode passMode)
    : maxDepth(maxDepth), logMode(logMode), passMode(passMode),
      enableHashPruning(enableHashPruning), cntAppliedRules(12) {}

int Derivator::getNumIntermediateStates() { return cntStates; }

void Derivator::guidedSearch(Formula &origin, int depth) {
    if (origin.root->getType() == NodeType::TensorNodeType) {
        auto tensor = as<TensorNode>(origin.root);
        appendCanddiate(tensor, depth);
        return;
    }
    Expr *expr = &origin.root;
    nnet_assert((*expr)->getType() == NodeType::RangeOpNodeType, __LINE__);
    while ((*expr)->getType() == NodeType::RangeOpNodeType) {
        auto rangeOp = as<RangeOpNode>(*expr);
        checkOOB(rangeOp);
        auto summand = rangeOp->getSummand();
        if (summand->getType() == NodeType::SubscriptNodeType) {
            auto subscriptOp = as<SubscriptNode>(summand);
            if (rule4StageMerging(origin, depth, *expr)) {
                return;
            }
            expr = subscriptOp->getObjectPtr();
            nnet_assert(*expr != nullptr, __LINE__);
            continue;
        }
        if (summand->getType() == NodeType::BinaryOpNodeType) {
            break;
        }
        nnet_unimplemented_halt();
    }

    if (searchState == 0) {
        searchState = 1;
        rule8GuidedDLT(origin, depth, *expr);
        searchState = 0;
        return;
    }
    if (searchState == 1) {
        rule8GuidedDLT(origin, depth, *expr);
        return;
    }
    if (searchState == 2) {
        matchComputationKernel(origin, depth, *expr);
        return;
    }
    if (searchState == 3) {
        // Pack the remaining computation as a MemBoundOp
        matchMemBoundKernel(origin, depth, origin.root);
        return;
    }
    nnet_unimplemented_halt();
    return;
}

void Derivator::ruleBasedDerivate(Formula &origin, int depth) {
    string StartDfs = "ruleBasedDerivate dep=" + std::to_string(depth) +
                      ", targetOp=" + std::to_string(routineTypeToId(targetOp));
    dbg(StartDfs, origin);
    auto tensor = as<TensorNode>(origin.root);
    if (tensor) {
        appendCanddiate(tensor, depth);
        return;
    }
    if (depth >= (int)rulesOverall.size())
        return;
    RangeOp curRangeOp;
    for (Expr *curExpr = &origin.root;
         curExpr && (curRangeOp = as<RangeOpNode>(*curExpr));) {
        checkOOB(curRangeOp);
        auto subscript = as<SubscriptNode>(curRangeOp->getSummand());
        // isSimplyNested: a directly nested stage
        bool isSimplyNested = (subscript && subscript->isRangeOpSubscripted());
        if (rule4StageMerging(origin, depth, *curExpr))
            return;
        // For the next nested stage
        curExpr = (isSimplyNested) ? subscript->getObjectPtr() : nullptr;
    }
    int stageDepth = 0;
    for (Expr *curExpr = &origin.root;
         curExpr && (curRangeOp = as<RangeOpNode>(*curExpr));) {
        // isSimplyNested: a directly nested stage
        auto subscript = as<SubscriptNode>(curRangeOp->getSummand());
        bool isSimplyNested = (subscript && subscript->isRangeOpSubscripted());
        stageDepth++;

        for (int rule : rulesOverall[depth]) {
            if (rule == 1)
                rule1VariableSplit(origin, depth, *curExpr);
            else if (!isSimplyNested) {
                if (rule == 2)
                    rule2VariableMerging(origin, depth, *curExpr);
                else if (rule == 3)
                    rule3StageSplit(origin, depth, *curExpr);
                else if (rule == 5)
                    rule5RangeRelaxation(origin, depth, *curExpr);
                else if (rule == 6)
                    rule6KenerlMatching(origin, depth, *curExpr);
                else if (rule == 7)
                    rule7DLT(origin, depth, *curExpr);
                else if (rule == 8)
                    rule8GuidedDLT(origin, depth, *curExpr);
                else if (rule == 9)
                    rule9RangeMagnify(origin, depth, *curExpr);
            }
        }
        // For the next nested stage
        curExpr = (isSimplyNested) ? subscript->getObjectPtr() : nullptr;
    }
    for (int rule : rulesOverall[depth])
        if (rule == 90 && stageDepth == 2) // HACK: for (T)Conv2gemm
            rule90TwoStageElementWise(origin, depth, origin.root);
        else if (rule == 91 && stageDepth >= 2) // HACK: for TConv2gemm
            rule91MergeStagesWithSum(origin, depth, origin.root);
}

void Derivator::nextStep(Formula &origin, int depth, Expr &rCur, Expr newCur) {
    // Count the number of searched states
    ++cntStates;
    rCur.swap(newCur);

    HashType formulaHash = HashVisitor().getHash(origin.root);
    if (enableHashPruning) {
        if (searchState != 2) {
            if (visited.find(formulaHash) != visited.end()) {
                rCur.swap(newCur);
                return;
            }
            visited.emplace(formulaHash);
        }
    }

    if (searchState > 0) {
        guidedSearch(origin, depth);
    } else {
        searchedMaxDepth = max(searchedMaxDepth, depth + 1);

        if (searchStrategy == Strategy::DFS ||
            (searchStrategy == Strategy::RuleAndDFS &&
             depth + 1 >= (ssize_t)rulesOverall.size()))
            dfs(origin, depth + 1);
        else
            ruleBasedDerivate(origin, depth + 1);
    }
    rCur.swap(newCur);
}

void Derivator::ruleBasedDFS(Formula &origin, int depth, vector<int> _rules,
                             map<int, vector<Iterator>> _substituteRules,
                             bool searchAfterRules) {
    SaveStateGuard guard(*this, origin.root, string("Init: ") + __FUNCTION__);
    searchStrategy = (searchAfterRules) ? Strategy::RuleAndDFS : Strategy::Rule;
    rulesOverall.clear();
    for (auto i : _rules)
        rulesOverall.push_back({i});
    substituteRules = _substituteRules;
    ruleBasedDerivate(origin, depth);
}

void Derivator::search(Formula &origin, int depth) {
    SaveStateGuard guard(*this, origin.root, string("Init: ") + __FUNCTION__);
    searchStrategy = Strategy::DFS;
    dfs(origin, depth);
}

void Derivator::print() {
    std::cout << "[RESULT] Derivator::results: " << candidates.size()
              << std::endl;
    std::cout << "==== DFS candidates (" << candidates.size()
              << ")====" << std::endl;
    for (const auto &f : candidates) {
        std::cout << f.toReadable() << std::endl;
        // dbg(f.bfsDepth, f.toReadable());
    }
    std::cout << "==== DFS log end ====" << std::endl;
}

string Formula::toReadable() const { return FullPrinterVisitor().print(root); }

void Derivator::rule1VariableSplit(Formula &origin, int depth, Expr &rCur) {
    ++cntAppliedRules[1];
    Rule1VariableSplit(*this).run(origin, depth, rCur);
    --cntAppliedRules[1];
}

void Derivator::rule2VariableMerging(Formula &origin, int depth, Expr &rCur) {
    ++cntAppliedRules[2];
    Rule2VariableMerging(*this).run(origin, depth, rCur);
    --cntAppliedRules[2];
}

void Derivator::rule3StageSplit(Formula &origin, int depth, Expr &rCur) {
    ++cntAppliedRules[3];
    Rule3StageSplit(*this).run(origin, depth, rCur);
    --cntAppliedRules[3];
}

bool Derivator::rule4StageMerging(Formula &origin, int depth, Expr &rCur,
                                  bool mergeStageWithCalc) {
    ++cntAppliedRules[4];
    Rule4StageMerging pass(*this);
    pass.setMergeStageWithCalc(mergeStageWithCalc);
    pass.run(origin, depth, rCur);
    --cntAppliedRules[4];
    return pass.isSuccessful();
}

void Derivator::rule5RangeRelaxation(Formula &origin, int depth, Expr &rCur) {
    ++cntAppliedRules[5];
    Rule5RangeRelaxation(*this).run(origin, depth, rCur);
    --cntAppliedRules[5];
}

void Derivator::rule6KenerlMatching(Formula &origin, int depth, Expr &rCur) {
    ++cntAppliedRules[6];
    Rule6KenerlMatching(*this).run(origin, depth, rCur);
    --cntAppliedRules[6];
}

void Derivator::rule7DLT(Formula &origin, int depth, Expr &rCur) {
    ++cntAppliedRules[7];
    Rule7DLT(*this).run(origin, depth, rCur);
    --cntAppliedRules[7];
}

void Derivator::rule8GuidedDLT(Formula &origin, int depth, Expr &rCur) {
    ++cntAppliedRules[8];
    Rule8GuidedDLT(*this).run(origin, depth, rCur);
    --cntAppliedRules[8];
}

void Derivator::rule9RangeMagnify(Formula &origin, int depth, Expr &rCur) {
    ++cntAppliedRules[9];
    Rule9RangeMagnify(*this).run(origin, depth, rCur);
    --cntAppliedRules[9];
}

void Derivator::rule90TwoStageElementWise(Formula &origin, int depth,
                                          Expr &rCur) {
    Rule90TwoStageElementWise(*this).run(origin, depth, rCur);
}

void Derivator::rule91MergeStagesWithSum(Formula &origin, int depth,
                                         Expr &rCur) {
    Rule91MergeStagesWithSum(*this).run(origin, depth, rCur);
}

void Derivator::matchComputationKernel(Formula &origin, int depth, Expr &rCur) {
    MatchComputationKernel(*this).run(origin, depth, rCur);
}

void Derivator::matchMemBoundKernel(Formula &origin, int depth, Expr &rCur) {
    MatchMemBoundKernel(*this).run(origin, depth, rCur);
}

bool Derivator::stageCombination(MultiFormulas &origin, int depth) {
    return (CompareMultiFormulasVisitor().compare(origin.roots));
}

Expr Derivator::mergeMemboundStages(VecExpr stages) {
    auto nested = MergeMemboundMutator(stages).merge();
    return nested;
}

void Derivator::appendCanddiate(const Tensor &tensor, int depth) {
    // if (!CountRoutineVisitor().match(tensor, 1, 0, 3))
    //     return;

    candidates.emplace_back(tensor, depth);
    // dbg("!!!!!!!!!!!!!!!Success!!!!!!!!!!!!!!!");
    // if (enableEquivalenceCheck)
    //     checkDerivationEquivalence();
    // printIntermediateStates();
    // puts("Success ===");
}

bool Derivator::checkOOB(const RangeOp &rangeOp, bool halt) {
    // Skip check in NoLog mode
    if (logMode == LogMode::NoLog)
        return false;
    bool hasOOB = CheckOOBVisitor().checkRangeOp(rangeOp);
    if (hasOOB) {
        printIntermediateStates();
        dbg(FullPrinterVisitor().print(rangeOp));
        if (halt)
            nnet_assert(0, "Out Of Bound in index!");
    }
    return hasOOB;
}

string Derivator::newTensorName() {
    return "T" + std::to_string(++nTensorNames);
}

Var Derivator::getNewVar() {
    return make_ref<VarNode>("i" + std::to_string(++nIteratorNames));
}

void Derivator::pushIntermediateState(const Expr &expr) {
    intermediateStates.emplace_back(CloneMutator().clone(expr));
};

void Derivator::pushRuleState(const string &state) {
    ruleStates.emplace_back(state);
}

void Derivator::pushRuleMsg(const string &state) {
    ruleMsgs.emplace_back(state);
    dbg(ruleMsgs.size(), ruleStates.size());
}

void Derivator::popIntermediateState() { intermediateStates.pop_back(); }

void Derivator::popRuleState() { ruleStates.pop_back(); }

void Derivator::popRuleMsg() { ruleMsgs.pop_back(); }

RoutineType Derivator::getTargetOp() { return targetOp; }

void Derivator::setTargetOp(RoutineType _targetOp) { targetOp = _targetOp; }

int Derivator::getSearchState() { return searchState; }

void Derivator::setSearchState(int _searchState) { searchState = _searchState; }

void Derivator::printStatistics() {
    printf("==== Derivator statistics ====\n");
    printf("Max Depth = %d\n", maxDepth);
    printf("searchStrategy = ");
    if (searchStrategy == Strategy::DFS)
        printf("DFS\n");
    else if (searchStrategy == Strategy::Rule)
        printf("Rule\n");
    else if (searchStrategy == Strategy::RuleAndDFS)
        printf("RuleAndDFS\n");
    printf("enableHashPruning = %s\n", enableHashPruning ? "true" : "false");
    printf("Reached Max Depth during search = %d\n", searchedMaxDepth);
    printf("#Candidates = %lu\n", candidates.size());
    printf("#Intermediate states = %d\n", cntStates);
    printf("#Hashed intermediate states = %lu\n", visited.size());
    printf("#Iteratos = %d\n", nIteratorNames);
    printf("#Tensors = %d\n", nTensorNames);
}

void Derivator::setDumpFirstSuccess(const string &_logFnPrefix) {
    setEquivalenceCheck();
    logMode = LogMode::DumpFristCandiate;
    logFnPrefix = _logFnPrefix;
}

void Derivator::printIntermediateStates() {
    // Skip in NoLog mode
    if (logMode == LogMode::NoLog)
        return;
    assert(intermediateStates.size() == ruleStates.size());
    assert(intermediateStates.size() == ruleMsgs.size());
    for (size_t i = 0; i < intermediateStates.size(); ++i) {
        string msg = "=== Depth " + std::to_string(i) + " " + ruleStates[i] +
                     ": " + ruleMsgs[i];
        std::cout << msg << endl;
        std::cout << FullPrinterVisitor().print(intermediateStates[i]) << endl;
        if (logMode == LogMode::DumpFristCandiate) {
            Serializer serializer;
            serializer.serialize(intermediateStates[i],
                                 logFnPrefix + to_string(i) + ".expr", msg);
        }
    }
    for (size_t i = 0; i < intermediateStates.size(); ++i) {
        if (auto cur = as<RangeOpNode>(intermediateStates[i]))
            if (CheckOOBVisitor().checkRangeOp(cur)) {
                printf("OOB detected depth=%lu\n", i);
            }
    }
    if (logMode == LogMode::DumpFristCandiate) {
        puts("Serializaiton finished.");
        exit(0);
    }
}

void Derivator::checkDerivationEquivalence() {
    if (!checkExprsEquvivalence(intermediateStates)) {
        nnet_assert(0, "Inequivalent derivation");
        exit(1);
    }
}

void Derivator::setEquivalenceCheck() { enableEquivalenceCheck = true; }

Derivator::PassMode Derivator::getPassMode() { return passMode; }

Derivator::LogMode Derivator::getLogMode() { return logMode; }

} // namespace nnet
