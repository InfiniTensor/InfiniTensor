#ifndef BANG_KERNELS_DIVOPERATION_DIV_H_
#define BANG_KERNELS_DIVOPERATION_DIV_H_


__mlu_global__ void MLUDivKernelUnion1(float *output,
                                       float *input1,
                                       float *input2,
                                       uint32_t num);

#endif //BANG_KERNELS_DIVOPERATION_DIV_H_

