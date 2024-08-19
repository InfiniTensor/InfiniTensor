#include "code_gen/nnet/Visitor/CheckOOBVisitor.h"
#include "code_gen/nnet/Visitor/SimplifyExprVisitor.h"

namespace nnet {

void CheckOOBVisitor::visit_(const Subscript &c) {
    const auto &objectRanges = c->getObjectRanges();
    for (size_t dim = 0; dim < c->getDims(); ++dim) {
        SimplifyExprVisitor simplifier;
        auto optional = simplifier.getExprRange(c->getIndex(dim), rangeOp);
        if (!optional.has_value())
            continue;
        const Range &exprRange = *optional;
        if (exprRange.first < objectRanges[dim].first ||
            exprRange.second > objectRanges[dim].second) {
            // dbg("OOB detected!", c, dim, exprRange, objectRanges[dim]);
            // std::cout << "OOB detected! " << c->toReadable() << ", dim=" <<
            // dim
            //           << ", Range=(" << exprRange.first << ", "
            //           << exprRange.second << "), objRange=("
            //           << objectRanges[dim].first << ", "
            //           << objectRanges[dim].second << ")." << std::endl;
            detect = true;
        }
    }
}

bool CheckOOBVisitor::checkRangeOp(const RangeOp &_rangeOp) {
    detect = false;
    rangeOp = _rangeOp;
    dispatch(rangeOp);
    return detect;
}

} // namespace nnet