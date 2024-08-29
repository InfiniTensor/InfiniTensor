#pragma once
#include "common.h"
#include "derivator.h"

namespace nnet {
#define DEFINE_VAR(name) auto name = make_ref<VarNode>(#name);
int matchExprResult(Derivator &derivator, string fn);
bool checkExprLogSame(string fnPrefix, int start, int end);
bool checkExprsEquvivalence(VecExpr exprs);
} // namespace nnet