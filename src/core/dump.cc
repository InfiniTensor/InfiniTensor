#include "core/dump.h"
#include "core/operator.h"

namespace infini {

void DumpObj::dumpOp(const Operator op) {
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

bool DumpObj::queriedOp(const Operator op, int count) {
    return strcmp(op->getOpType().toString(), opKey.c_str()) == 0 &&
           location == count;
}

} // namespace infini
