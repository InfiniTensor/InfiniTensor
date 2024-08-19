#include "code_gen/nnet/Visitor/CountRoutineVisitor.h"

namespace nnet {

void CountRoutineVisitor::visit_(const Tensor &c) {
    if (auto routine = c->getSource(); routine) {
        cnts[routineTypeToId(routine->getType())]++;
    }
    ExprTreeVisitor::visit_(c);
}

vector<int> CountRoutineVisitor::count(const Expr &root) {
    cnts = vector<int>(RoutineTypeCnt, 0);
    dispatch(root);
    return cnts;
}

bool CountRoutineVisitor::match(const Expr &root, int nMatmul, int nConv,
                                int nElement, int nSg2bmm,
                                int nLongformerGBMM) {
    auto opCount = count(root);
    bool ret = true;
    if (opCount[routineTypeToId(RoutineType::MatmulNodeType)] != nMatmul)
        ret = false;
    if (opCount[routineTypeToId(RoutineType::ConvNodeType)] != nConv)
        ret = false;
    if (opCount[routineTypeToId(RoutineType::ElementWiseNodeType)] != nElement)
        ret = false;
    if (opCount.at(routineTypeToId(RoutineType::G2bmmNodeType)) != nSg2bmm)
        ret = false;
    if (!ret) {
        auto target =
            vector<int>{nMatmul, nConv, nSg2bmm, nLongformerGBMM, nElement};
    }
    return ret;
}

} // namespace nnet