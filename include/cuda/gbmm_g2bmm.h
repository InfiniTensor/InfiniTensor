#pragma once
#include <cuda_fp16.h>

namespace infini {

void _sg2bmm(float *__restrict__ q, float *__restrict__ k,
             float *__restrict__ y, int bs, int n, int m, int w, int d);

void _sgbmml(float *__restrict__ q, float *__restrict__ k,
             float *__restrict__ y, int bs, int n, int m, int w, int d);

void _sg2bmm(const half *q, const half *k, half *y, int bs, int n, int m,
              int w, int d);
void _sgbmml(const half *q, const half *k, half *y, int bs, int n, int m,
              int w, int d);

} // namespace infini
