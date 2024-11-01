#include "code_gen/generator.h"
#include "code_gen/graph.h"
#include "code_gen/operator.h"
#include "code_gen/search_engine.h"
#include "code_gen/tensor.h"
#include "code_gen/perf_engine.h"
#include <cstdio>
#include <iostream>
#include "test.h"

const int kh = 3, kw = 3,  dh = 1,  dw = 1,  ph = 0,  pw = 0,
           sh = 2,  sw = 2;

TEST(PERF_POOL, Cuda_codeGenerate) {
    tpm::PerfEngine pe;
    auto i0 = tpm::Tensor({1, 64, 162, 162});
    auto maxpool = tpm::MaxPoolOp(&i0, kh, kw, dh, dw, ph, pw, sh, sw);
    auto avgpool = tpm::AvgPoolOp(&i0, kh, kw, ph, pw, sh, sw);
    printf("Maxpool perf %lf ms\n", maxpool.perf(&pe));
    printf("Avgpool perf %lf ms\n", avgpool.perf(&pe));
}
