#pragma once
#include "common.h"
#include "derivator.h"
#include "gtest/gtest.h"

// clang-format off
#define CAT(A, B) A##B
#define SELECT(NAME, NUM) CAT(NAME##_, NUM)
#define GET_COUNT( _1, _2, _3, _4, _5, _6, _7, _8, _9, _10, COUNT, ... ) COUNT
#define VA_SIZE( ... ) GET_COUNT( __VA_ARGS__, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1 )
#define VA_SELECT( NAME, ... ) SELECT( NAME, VA_SIZE(__VA_ARGS__) )(__VA_ARGS__)

#define _DEFVAR_1(name) auto name = make_ref<VarNode>(#name);
#define _DEFVAR_2(name, ...) _DEFVAR_1(name); _DEFVAR_1(__VA_ARGS__)
#define _DEFVAR_3(name, ...) _DEFVAR_1(name); _DEFVAR_2(__VA_ARGS__)
#define _DEFVAR_4(name, ...) _DEFVAR_1(name); _DEFVAR_3(__VA_ARGS__)
#define _DEFVAR_5(name, ...) _DEFVAR_1(name); _DEFVAR_4(__VA_ARGS__)
#define _DEFVAR_6(name, ...) _DEFVAR_1(name); _DEFVAR_5(__VA_ARGS__)
#define _DEFVAR_7(name, ...) _DEFVAR_1(name); _DEFVAR_6(__VA_ARGS__)
#define _DEFVAR_8(name, ...) _DEFVAR_1(name); _DEFVAR_7(__VA_ARGS__)
#define _DEFVAR_9(name, ...) _DEFVAR_1(name); _DEFVAR_8(__VA_ARGS__)
#define _DEFVAR_10(name, ...) _DEFVAR_1(name); _DEFVAR_9(__VA_ARGS__)
#define DEFINE_VAR(...) VA_SELECT(_DEFVAR, __VA_ARGS__)
// clang-format on

namespace nnet {
int matchExprResult(Derivator &derivator, string fn);
bool checkExprLogSame(string fnPrefix, int start, int end);
bool checkExprsEquvivalence(VecExpr exprs);
} // namespace nnet
