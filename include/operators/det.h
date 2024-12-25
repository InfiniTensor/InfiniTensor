#pragma once
#include "core/operator.h"

namespace infini {
class DetObj : public OperatorObj {
  public:
    enum Mode { NormalDet = 0, LogDet };
    static Mode fromModeStr(const std::string &mode) {
        if (mode == "normal") {
            return NormalDet;
        } else if (mode == "log") {
            return LogDet;
        } else {
            IT_TODO_HALT();
        }
    }
    DetObj(GraphObj *graph, Tensor input, Tensor output,
           const std::string &mode);
    OP_CLONE(DetObj);
    optional<vector<Shape>> inferShape(const TensorVec &inputs) override;

    std::string toString() const override;
    int numInputs() const override { return 1; }
    int numOutputs() const override { return 1; }
    Mode getMode() const { return modeValue; }
    std::string getModeStr() const;
    Mode strToMode(const std::string &modeStr) const;

  private:
    Mode modeValue;
    vector<int> getWorkloadVector() const override;
    vector<int> getOpAttrVector() const override;
};
}; // namespace infini
