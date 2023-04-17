#include "core/operator.h"

namespace infini {
class UnsqueezeObj : public OperatorObj {
    set<int> axis;

  public:
    UnsqueezeObj(GraphObj *graph, Tensor input, const vector<int> &axis,
                 Tensor output);
    OP_CLONE(UnsqueezeObj);

    optional<vector<Shape>> inferShape(const TensorVec &inputs) const override;
    std::string toString() const override;
    int numInputs() const override { return 1; }
    int numOutputs() const override { return 1; }

  private:
    bool parseAxis(const std::vector<int> &index, std::set<int> &axis) const;
    vector<int> getWorkloadVector() const override;
    vector<int> getOpAttrVector() const override;
};

} // namespace infini
