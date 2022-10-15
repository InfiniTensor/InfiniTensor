#include "cuda/gbmm_g2bmm.cuh"
#include "cuda/gbmm_g2bmm.h"

namespace infini {

void _sg2bmm(float *__restrict__ q, float *__restrict__ k,
             float *__restrict__ y, int bs, int n, int m, int w, int d) {
    sg2bmm(q, k, y, bs, n, m, w, d);
}

void _sgbmml(float *__restrict__ q, float *__restrict__ k,
             float *__restrict__ y, int bs, int n, int m, int w, int d) {
    sgbmml(q, k, y, bs, n, m, w, d);
}

} // namespace infini
