#include "code_gen/nnet/Pass/Pass.h"
#include "code_gen/nnet/Visitor/CloneMutator.h"

namespace nnet {

Pass::Pass(Derivator &derivator, const string &passName)
    : derivator(derivator), passName(passName),
      enableLogging(derivator.getLogMode() != Derivator::LogMode::NoLog),
      enableDebug(false) {}

Pass::~Pass() = default;

void Pass::setEnableLogging(bool value) { enableLogging = value; }

void Pass::setEnableDebug(bool value) { enableDebug = value; }

void Pass::run(Formula &origin, int dfsDepth, Expr &rCur) {
    initialize(origin, rCur);
    transform(origin, dfsDepth, rCur);
    finalize();
}

void Pass::initialize(Formula &origin, const Expr &rCur) {}

void Pass::finalize() {}

Var Pass::getNewVar() { return derivator.getNewVar(); }

string Pass::newTensorName() { return derivator.newTensorName(); }

void Pass::nextStep(Formula &origin, int depth, Expr &rCur, Expr newCur,
                    const string &ruleMsg) {
    // push rule action description
    if (enableLogging) {
        rCur.swap(newCur);
        derivator.pushIntermediateState(origin.root);
        rCur.swap(newCur);
        derivator.pushRuleState(passName);
        derivator.pushRuleMsg(ruleMsg);
    }

    if (enableDebug) {
        // In debug mode, do not recur but save the transformed state
        transformations.emplace_back(CloneMutator().clone(newCur));
    } else
        derivator.nextStep(origin, depth, rCur, newCur);

    // pop rule action description
    if (enableLogging) {
        derivator.popIntermediateState();
        derivator.popRuleState();
        derivator.popRuleMsg();
    }
}

const VecExpr &Pass::getTransformations() { return transformations; }

} // namespace nnet