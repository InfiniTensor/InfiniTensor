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

int main(int argc, char **argv) {
    int n = 64, c = 256, h = 14, w = 14, f = 256, wc = 256;
    if (argc == 7) {
        n = atoi(argv[1]);
        c = atoi(argv[2]);
        h = atoi(argv[3]);
        w = atoi(argv[4]);
        f = atoi(argv[5]);
        wc = atoi(argv[6]);
    } else if (argc == 6) {
        n = atoi(argv[1]);
        c = atoi(argv[2]);
        h = atoi(argv[3]);
        w = atoi(argv[4]);
        f = atoi(argv[5]);
    } else if (argc == 5) {
        n = atoi(argv[1]);
        c = atoi(argv[2]);
        h = atoi(argv[3]);
        w = h;
        f = atoi(argv[4]);
        wc = c;
    } else if (argc > 1) {
        std::cout << "Arg formats:" << std::endl;
        std::cout << "./bin n c h w f wc" << std::endl;
        std::cout << "./bin n c h w f" << std::endl;
        std::cout << "./bin n c insize f" << std::endl;
        return -1;
    }

    tpm::Graph g{};
    tpm::PerfEngine pe{};
    auto b = c / wc;
    auto i0 = g.tensor({b, n * h * w, c});
    auto w0 = g.tensor({b, wc, f / b});
    auto gemm = g.matmul(i0, w0);
    auto i1 = gemm->getOutputs()[0];
    auto outDim = i1->getDims();
    std::cout << "Conv: input = [ " << n << ", " << c << ", " << h << ", " << w
              << "], weight = [" << f << ", " << wc << ", 1, 1"
              << "], "
              << "], output = " << tpm::dimToString(outDim) << std::endl;

    std::cout << gemm->perf(&pe, 10, 2) << std::endl;

    // i0->dataRand();
    // i1->dataMalloc();
    // w0->dataRand();

    // struct timeval beg, end;
    // gettimeofday(&beg, 0);
    // op0->compute();
    // gettimeofday(&end, 0);
    // std::cout << "conv time: " << getDurtime(beg, end) << std::endl;

    return 0;
}
