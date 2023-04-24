#pragma once

namespace infini {

void conv2dreduce_kernel(float *input, float *bias, float *output, bool PReLU,
                         int n, int h, int w, int f, int r, int s, int oh,
                         int ow, int ph, int pw, int sh, int sw, int dh,
                         int dw);

void convTranspose2dreduce_kernel(float *input, float *bias, float *output,
                                  int act, int n, int h, int w, int f, int r,
                                  int s, int oh, int ow, int ph, int pw, int sh,
                                  int sw, int dh, int dw);

void reduceConvRxSToNCHW(float *input, float *bias, float *output, int act,
                         int n, int h, int w, int f, int r, int s, int oh,
                         int ow, int ph, int pw, int sh, int sw, int dh,
                         int dw);

void convTranspose2dreduce_kernel(float *input, float *bias, float *output,
                                  int act, int n, int h, int w, int f, int r,
                                  int s, int oh, int ow, int ph, int pw, int sh,
                                  int sw, int dh, int dw);

void conv5x5ToConv3x3Reduce(int n, int f, int h, int w, float *input,
                            float *output, float *bias);

void conv3x3ToReduce(int n, int h, int w, int f, float *input, float *output,
                     float *bias);

} // namespace infini
