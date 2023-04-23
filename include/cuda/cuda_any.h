#pragma once

#include "operators/any.h"

namespace infini {

void any_kernel_mapping(vector<float *> input, vector<float *> output,
                        const string &kernel_name, const vector<int> &attr);

} // namespace infini
