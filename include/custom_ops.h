#ifndef CUSTOM_OPS_H
#define CUSTOM_OPS_H

namespace infini {

void _sg2bmm(float *__restrict__ q, float *__restrict__ k,
             float *__restrict__ y, int bs, int n, int m, int w, int d);

void _sgbmml(float *__restrict__ q, float *__restrict__ k,
             float *__restrict__ y, int bs, int n, int m, int w, int d);

} // namespace infini

#endif // CUSTOM_OPS_H
