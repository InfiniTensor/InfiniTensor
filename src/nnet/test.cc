#include "nnet/Visitor/FullPrinterVisitor.h"
#include "nnet/Visitor/GetTensorsVisitor.h"
#include "nnet/Visitor/HashVisitor.h"
#include "nnet/Visitor/Interpreter.h"
#include "nnet/Visitor/Serializer.h"
#include <filesystem>
namespace nnet {

std::filesystem::path getProjectHome() {
#ifndef INFINI_PROJECT_HOME
#error INFINI_PROJECT_HOME is not defined
#endif

#define Q(x) #x
#define QUOTE(x) Q(x)
#define PROJECT_HOME QUOTE(INFINI_PROJECT_HOME)
    return std::filesystem::path(PROJECT_HOME);
#undef PROJECT_HOME
#undef QUOTE
#undef Q
}

string getResourceFilePath(string path) {
    return (getProjectHome() / path).string();
}

int matchExprResult(Derivator &derivator, string pathRelativeToProjectHome) {
    auto fullPath = getResourceFilePath(pathRelativeToProjectHome);
    auto ans = Serializer().deserialize(fullPath);
    auto hashAns = HashVisitor()(ans);
    int match = 0;
    for (const auto &candidate : derivator.getCandidates()) {
        auto hashCandidate = HashVisitor()(candidate.root);
        match += (hashAns == hashCandidate);
    }
    return match;
}

bool checkExprLogSame(string pathRelativeToProjectHome, int start, int end) {
    auto fullPath = getResourceFilePath(pathRelativeToProjectHome);
    Serializer serializer;
    string fn0 = fullPath + to_string(start) + ".expr";
    Expr expr0 = serializer.deserialize(fn0);
    RangeOp range0 = as<RangeOpNode>(expr0);
    Interpreter interpreter(range0);
    auto ans0 = interpreter.interpretUniformSample(range0);
    dbg(expr0, ans0);
    for (int i = start + 1; i < end; ++i) {
        string fn1 = fullPath + to_string(i) + ".expr";
        Expr expr1 = serializer.deserialize(fn1);
        RangeOp range1 = as<RangeOpNode>(expr1);
        dbg(fn1, expr1);
        auto ans1 = interpreter.interpretUniformSample(range1);
        dbg(ans1);
        if (ans0.size() != ans1.size())
            return false;
        for (size_t i = 0; i < ans0.size(); ++i)
            if (ans0[i] != ans1[i])
                return false;
    }
    return true;
}

bool checkExprsEquvivalence(VecExpr exprs) {
    if (exprs.size() < 2)
        return true;
    auto inputsMap0 = GetTensorsVisitor().get(exprs[0]);
    RangeOp range0 = as<RangeOpNode>(exprs[0]);
    Interpreter interpreter(range0);
    auto ans0 = interpreter.interpretUniformSample(range0);
    for (size_t i = 1; i + 1 < exprs.size(); ++i) {
        RangeOp range1 = as<RangeOpNode>(exprs[i]);
        auto inputsMap1 = GetTensorsVisitor().get(range1);
        // if expr0 and expr1 have different inputs, skip and return true
        if (inputsMap0.size() != inputsMap1.size())
            return true;
        for (const auto &[name, tensor] : inputsMap0) {
            if (!inputsMap1.count(name))
                return true;
        }
        auto ans1 = interpreter.interpretUniformSample(range1);
        if (ans0.size() != ans1.size())
            return false;
        for (size_t i = 0; i < ans0.size(); ++i)
            if (ans0[i] != ans1[i])
                return false;
    }
    return true;
}

} // namespace nnet
