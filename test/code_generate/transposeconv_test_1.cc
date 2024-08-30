#include "code_gen/generator.h"
#include "code_gen/graph.h"
#include "code_gen/operator.h"
#include "code_gen/perf_engine.h"
#include "code_gen/search_engine.h"
#include "code_gen/tensor.h"
#include <cstdlib>
#include <iostream>
#include <sys/time.h>

double getDurtime(struct timeval beg, struct timeval end) {
    double t =
        (1000000.0 * (end.tv_sec - beg.tv_sec) + end.tv_usec - beg.tv_usec) /
        1000.0;
    return t;
}

const int n = 1, c = 256, h = 2, w = 2, f = 448, r = 4, s = 4;

int main() {
    auto i0 = new tpm::Tensor({n, h, w, f});
    auto w0 = new tpm::Tensor({f, r, s, c});
    auto i1 = new tpm::Tensor({n, h, w, c});
    auto op0 = new tpm::ConvTransOp(i0, w0, i1, 0, 0, 1, 1, 1, 1);

    i0->dataRand();
    i1->dataMalloc();
    w0->dataRand();
    auto pe = new tpm::PerfEngine();
    struct timeval beg, end;
    gettimeofday(&beg, 0);
    op0->perf(pe, 1000, 100);
    gettimeofday(&end, 0);
    std::cout << "conv time: " << getDurtime(beg, end) << std::endl;

    delete i0;
    delete i1;
    delete w0;
    delete op0;

    return 0;
}
