#pragma once
#include "core/tensor.h"
#include "cuda/cuda_common.h"

namespace infini {

void cudaPrintFloat(float *x, int len);

void cudaPrintTensor(const Tensor &tensor);

cudnnDataType_t cudnnDataTypeConvert(DataType dataType);
cudaDataType cublasDataTypeConvert(DataType);

template <int index> struct DT_CUDA {};
template <> struct DT_CUDA<0> { using t = bool; };
template <> struct DT_CUDA<1> { using t = float; };
template <> struct DT_CUDA<2> { using t = unsigned char; };
template <> struct DT_CUDA<3> { using t = char; };
template <> struct DT_CUDA<4> { using t = unsigned short; };
template <> struct DT_CUDA<5> { using t = short; };
template <> struct DT_CUDA<6> { using t = int; };
template <> struct DT_CUDA<7> { using t = long long; };
template <> struct DT_CUDA<9> { using t = bool; };
template <> struct DT_CUDA<10> { using t = half; };
template <> struct DT_CUDA<11> { using t = double; };
template <> struct DT_CUDA<12> { using t = unsigned int; };
template <> struct DT_CUDA<13> { using t = unsigned long long; };
template <> struct DT_CUDA<16> { using t = nv_bfloat16; };
} // namespace infini
