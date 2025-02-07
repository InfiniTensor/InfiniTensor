#pragma once
#include "core/operator.h"
#include <algorithm>

namespace infini {
/**
 * @brief Produce a slice of the input tensor along given dimensions.
 *
 */
class SliceObj : public OperatorObj {
    template <class T> struct range_t { T start, end, step; };
    vector<range_t<int>> axes;

  public:
    /**
     * @brief Construct a new Slice object.
     *
     * @param graph The computation graph that this operator belongs to.
     * @param input The input tensor.
     * @param output The output tensor.
     * @param starts The start position to slice at certain axes. `starts` is a
     * list which has the same length with axis.
     * @param ends The end position to slice at certain axes. `ends` is a list
     * which has the same length with axis.
     * @param axes The dimensions to slice. If `axis` is empty, it is set to [0,
     * 1, ..., d-1], where d is the number of dimensions of the input tensor.
     * @param steps The step to slice at certain axes. `step` is a list which
     * has the same length with axis.
     */
    SliceObj(GraphObj *graph, Tensor input, Tensor output,
             const vector<int> &starts, const vector<int> &ends,
             const optional<vector<int>> &axes,
             const optional<vector<int>> &steps);
    OP_CLONE(SliceObj);

    optional<vector<Shape>> inferShape(const TensorVec &inputs) override;
    std::string toString() const override;
    inline int numInputs() const override { return 1; }
    inline int numOutputs() const override { return 1; }
    inline Shape getStarts() const {
        Shape ans(axes.size());
        std::transform(axes.begin(), axes.end(), ans.begin(),
                       [](auto x) { return x.start; });
        return ans;
    }
    inline Shape getEnds() const {
        Shape ans(axes.size());
        std::transform(axes.begin(), axes.end(), ans.begin(),
                       [](auto x) { return x.end; });
        return ans;
    }
    inline Shape getSteps() const {
        Shape ans(axes.size());
        std::transform(axes.begin(), axes.end(), ans.begin(),
                       [](auto x) { return x.step; });
        return ans;
    }

  private:
    vector<int> getWorkloadVector() const override;
    vector<int> getOpAttrVector() const override;
};
} // namespace infini
