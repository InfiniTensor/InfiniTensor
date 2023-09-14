#include "core/dump.h"

namespace infini {

void Dump::dumpOp(Operator op) {
    inputs.clear();
    outputs.clear();
    // Clone the inputs and outputs to host and store in dump
    for (Tensor input : op->getInputs()) {
        inputs.push_back(input->clone(NativeCpuRuntimeObj::getInstance()));
    }
    for (Tensor output : op->getOutputs()) {
        outputs.push_back(output->clone(NativeCpuRuntimeObj::getInstance()));
    }
}

bool Dump::queriedOp(Operator op, int count) {
    return strcmp(op->getOpType().toString(), opKey.c_str()) == 0 &&
           location == count;
}

} // namespace infini
