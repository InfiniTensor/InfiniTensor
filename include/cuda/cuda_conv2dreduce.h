#pragma once
#include "operators/conv2dreduce.h"

namespace infini {

void conv2dreduce_kernel(float *input, float *bias, float *output, bool PReLU,
                         int n, int h, int w, int f, int r, int s, int oh,
                         int ow, int ph, int pw, int sh, int sw, int dh,
                         int dw);

void convTranspose2dreduce_kernel(float *input, float *bias, float *output,
                                  bool PReLU, int n, int h, int w, int f, int r,
                                  int s, int oh, int ow, int ph, int pw, int sh,
                                  int sw, int dh, int dw);
} // namespace infini