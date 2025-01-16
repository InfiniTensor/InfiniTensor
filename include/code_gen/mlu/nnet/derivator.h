#pragma once
#include "common.h"
#include "expr.h"
#include "iterator_table.h"
#include "routine.h"
#include <iostream>
#include <sstream>
#include <unordered_set>

namespace nnet {

class Formula {
  public:
    Expr root;
    const int bfsDepth;

  public:
    Formula(Expr _root, int _bfsDepth) : root(_root), bfsDepth(_bfsDepth) {}
    string toReadable() const;
    friend std::ostream &operator<<(std::ostream &ios, const Formula &expr) {
        ios << expr.toReadable();
        return ios;
    }
    bool isVariable() const { return as<VarNode>(root) != nullptr; }
};

class MultiFormulas {
  public:
    VecExpr roots;
    const int bfsDepth;

  public:
    MultiFormulas(VecExpr roots, int _bfsDepth)
        : roots(roots), bfsDepth(_bfsDepth) {}
    // string toReadable() const;
    // friend std::ostream &operator<<(std::ostream &ios, const Formula &expr) {
    //     ios << expr.toReadable();
    //     return ios;
    // }
};

class Derivator {
  public:
    enum class LogMode { Normal, DumpFristCandiate, NoLog };
    enum class PassMode { Debug, Full };

  private:
    list<Formula> candidates;
    const int maxDepth;
    int nIteratorNames = 0;
    int nTensorNames = 0;
    vector<vector<int>> rulesOverall;
    enum class Strategy { DFS, Rule, RuleAndDFS } searchStrategy;
    LogMode logMode;
    PassMode passMode;
    bool enableEquivalenceCheck = false;
    string logFnPrefix;
    const bool enableHashPruning;
    int searchedMaxDepth = 0;
    RoutineType targetOp = RoutineType::NoneType;
    map<int, vector<Var>> substituteRules;

    vector<int> cntAppliedRules;
    int cntRule3 = 0;
    std::unordered_set<HashType> visited;
    VecExpr intermediateStates;
    vector<string> ruleStates, ruleMsgs;
    int cntStates = 0;   // the number of intermediate states
    int searchState = 0; // search state in guided search

  public:
    Derivator(int maxDepth = 8, bool enableHashPruning = true,
              LogMode mode = LogMode::Normal,
              PassMode passMode = PassMode::Debug);
    void search(Formula &origin, int depth);
    void ruleBasedDFS(Formula &origin, int depth, vector<int> _rules,
                      map<int, vector<Var>> _substituteRules = {},
                      bool searchAfterRules = false);
    void guidedSearch(Formula &origin, int depth);
    void print();
    int getNumCandidates() const { return candidates.size(); }
    const auto &getCandidates() const { return candidates; }
    void appendCanddiate(const Tensor &tensor, int depth);
    int getSearchedMaxDepth() const { return searchedMaxDepth; };
    bool stageCombination(MultiFormulas &origin, int depth);
    bool checkOOB(const RangeOp &rangeOp, bool halt = true);

    string newTensorName();
    Var getNewVar();

    Expr mergeMemboundStages(VecExpr stages);

  private:
    void dfs(Formula &origin, int depth);
    void ruleBasedDerivate(Formula &origin, int depth);

    void rule1VariableSplit(Formula &origin, int depth, Expr &rCur);
    void rule2VariableMerging(Formula &origin, int depth, Expr &rCur);
    void rule3StageSplit(Formula &origin, int dfsDepth, Expr &rCur);
    void rule5RangeRelaxation(Formula &origin, int depth, Expr &rCur);
    bool rule4StageMerging(Formula &origin, int depth, Expr &rCur,
                           bool mergeStageWithCalc = false);
    void rule6KenerlMatching(Formula &origin, int depth, Expr &rCur);
    void rule7DLT(Formula &origin, int depth, Expr &rCur);
    // Rule 8: guidedDLT
    void rule8GuidedDLT(Formula &origin, int depth, Expr &rCur);
    void rule9RangeMagnify(Formula &origin, int depth, Expr &rCur);
    void rule90TwoStageElementWise(Formula &origin, int depth, Expr &rCur);
    void rule91MergeStagesWithSum(Formula &origin, int depth, Expr &rCur);
    /**
     * @brief For searchState=2, wrap the RangeOp to add offset, if the boundary
     * does not start from 0. Then match the inner offset RangeOp.
     */
    void matchComputationKernel(Formula &origin, int depth, Expr &rcur);
    /**
     * @brief For searchState=3, the Formula must be a MemBound kernel?
     */
    void matchMemBoundKernel(Formula &origin, int depth, Expr &rcur);

    /**
     * @brief Check the equivalence for exprs in intermediateStates.
     */
    void checkDerivationEquivalence();

  public:
    void pushIntermediateState(const Expr &expr);
    void pushRuleState(const string &state);
    void pushRuleMsg(const string &state);
    void popIntermediateState();
    void popRuleState();
    void popRuleMsg();
    // void pushTransformInfo(const Expr &expr, const string &state,
    //                        const string &msg);
    void nextStep(Formula &origin, int depth, Expr &rCur, Expr newCur);

    RoutineType getTargetOp();
    void setTargetOp(RoutineType _targetOp);

    int getSearchState();
    void setSearchState(int _searchState);
    int getNumIntermediateStates();
    void printStatistics();
    void printIntermediateStates();
    void setDumpFirstSuccess(const string &_logFnPrefix);
    void setEquivalenceCheck();
    PassMode getPassMode();
    LogMode getLogMode();
};

} // namespace nnet
