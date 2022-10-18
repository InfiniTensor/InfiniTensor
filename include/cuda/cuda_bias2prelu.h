#pragma once
#include "operators/bias2prelu.h"

namespace infini {

void bias2prelu_kernel(float *input, float *bias, float *output, int n, int h,
                       int w, int c, bool PReLU, float paramPReLU);

}