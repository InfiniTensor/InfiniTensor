#pragma once
#include "cnnl.h"
namespace infini {
void TopKUnion_f32(cnnlHandle_t handle, float const *source, int64_t topk,
                   int64_t *Indices, float *Values, int othersize, int dimsize,
                   int Largest, int sorted);
void TopKUnion_f16(cnnlHandle_t handle, uint16_t const *source, int64_t topk,
                   int64_t *Indices, uint16_t *Values, int othersize,
                   int dimsize, int Largest, int sorted);
}; // namespace infini
