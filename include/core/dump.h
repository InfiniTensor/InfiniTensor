#pragma once

#include "core/common.h"
#include "core/operator.h"
#include "core/runtime.h"
#include "core/tensor.h"
/**
 * A Dump stores intermediate states of a model run, and exposes the info to
 * outside queries.
 */
namespace infini {
class Dump {
  protected:
    string opKey;
    int location = 0;
    vector<Tensor> inputs;
    vector<Tensor> outputs;

  public:
    Dump() {}

    /*
     * Dump the info of a operator.
     */
    void dumpOp(Operator op);

    vector<Tensor> getInputs() { return inputs; }
    vector<Tensor> getOutputs() { return outputs; }

    // TODO: For now, use type name and count to locate a specific operator.
    // In the future, use unique name or id of the queried operator.
    void setOpQuery(string opKey, int location) {
        this->opKey = opKey;
        this->location = location;
    }

    string getOpKey() { return this->opKey; }

    /*
     * True if op is queried.
     */
    bool queriedOp(Operator op, int count = 0);
};
} // namespace infini
