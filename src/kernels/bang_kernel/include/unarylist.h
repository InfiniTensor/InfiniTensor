#pragma once
__mlu_global__ void MLUUnaryKernelUnion1(float *output, float *input,
                                         uint32_t num, uint32_t op_list);
