#ifndef BANG_KERNELS_SOFTMAXOPERATION_SOFTMAX_H_
#define BANG_KERNELS_SOFTMAXOPERATION_SOFTMAX_H_

__mlu_global__ void softmaxUnion1(float *mlu_destination, float *mlu_src,
                                  int nDim, int axis, int othersize,
                                  int frontsize, int dimsize, int stride);

#endif // BANG_KERNELS_SOFTMAXOPERATION_SOFTMAX_H_
