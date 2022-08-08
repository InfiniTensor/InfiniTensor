#pragma once
#include "common.h"
#include "expr.h"
#include <iostream>

namespace nnet {

// enum class DLTType { Split, Merge, Reorder };

struct DLTOperation {
    // DLTType type;
    virtual ~DLTOperation() {}
};
struct DLTSplit : DLTOperation {
    int dim, factor;
    DLTSplit(int _dim, int _factor) : dim(_dim), factor(_factor) {}
};
struct DLTMerge : DLTOperation {
    int dim0, dim1;
    DLTMerge(int _dim0, int _dim1) : dim0(_dim0), dim1(_dim1) {}
};
struct DLTReorder : DLTOperation {
    vector<int> dims;
    DLTReorder(vector<int> _dims) : dims(_dims) {}
};

class DLT {
    vector<Ref<DLTOperation>> ops;

  public:
    /**
     * @brief dim -> (dim/factor, factor)
     */
    void split(int dim, int factor);
    /**
     * @brief Merge dim1 into dim0 -> (dim0, dim1)
     */
    void merge(int dim0, int dim1);
    /**
     * @brief
     *
     * @param dims dims[new_dim]=old_dim
     */
    void reorder(vector<int> dims);
    optional<Expr> apply(const RangeOp &rangeOp, const Subscript &subscript,
                         string newTensorName);

  private:
    optional<pair<Expr, Expr>> splitIndex(Expr expr, int factor,
                                          RangeOp rangeOp);
};

} // namespace nnet