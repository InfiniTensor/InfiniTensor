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
    /// One range per input dimension, worked out afresh whenever shapes are
    /// inferred: a range is only a pair of numbers once the input has a shape,
    /// and both the window it names and the extent of every dimension it leaves
    /// alone move with that shape.
    vector<range_t<int>> axes;
    /// The request as the model made it, kept because that is what survives a
    /// change of shape. A start of -1 means the last element whatever the
    /// length is, and an end past the end means the length itself; resolving
    /// either one against a shape that has since been replaced would answer for
    /// the wrong shape.
    vector<int64_t> starts, ends;
    optional<vector<int>> namedAxes, namedSteps;
    /// Whether the bounds are read from edges of the graph rather than from the
    /// request above.
    bool boundsFromEdges = false;

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
    /**
     * @brief Construct a new Slice object whose bounds are edges of the graph
     * rather than constants.
     *
     * An exporter writes a slice of a dynamic input's shape as a computation on
     * that shape, so the model holds no constant to read: the bounds are only
     * numbers once a shape is given. `starts` and `ends` carry them as shape
     * values, which `inferShape` reads afresh each time.
     *
     * @param graph The computation graph that this operator belongs to.
     * @param input The input tensor.
     * @param starts The rank one tensor holding the start of each range.
     * @param ends The rank one tensor holding the end of each range.
     * @param output The output tensor.
     * @param axes The dimensions to slice, which an exporter writes as a
     * constant even where the bounds are computed.
     * @param steps The step to slice at certain axes.
     */
    SliceObj(GraphObj *graph, Tensor input, Tensor starts, Tensor ends,
             Tensor output, const optional<vector<int>> &axes,
             const optional<vector<int>> &steps);
    OP_CLONE(SliceObj);

    optional<vector<Shape>> inferShape(const TensorVec &inputs) override;
    void inferShapeValue() override;
    std::string toString() const override;
    inline int numInputs() const override { return inputs.size(); }
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
    /// Work out one range per dimension of `shape` from the request, replacing
    /// whatever was worked out for an earlier shape.
    void resolveRanges(const Shape &shape);
    vector<int> getWorkloadVector() const override;
    vector<int> getOpAttrVector() const override;
};
} // namespace infini
