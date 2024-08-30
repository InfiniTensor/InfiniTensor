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
    int b = 1, m = 64, n = 1024, k = 1024;
    bool transA = false, transB = false;
    if (argc == 4) {
        m = atoi(argv[1]);
        n = atoi(argv[2]);
        k = atoi(argv[3]);
    } else if (argc == 5) {
        b = atoi(argv[1]);
        m = atoi(argv[2]);
        n = atoi(argv[3]);
        k = atoi(argv[4]);
    } else if (argc == 7) {
        b = atoi(argv[1]);
        m = atoi(argv[2]);
        n = atoi(argv[3]);
        k = atoi(argv[4]);
        transA = atoi(argv[5]) == 0 ? false : true;
        transB = atoi(argv[6]) == 0 ? false : true;
    } else {
        std::cout << "Arg formats:" << std::endl;
        std::cout << "./bin b m n k transA transB" << std::endl;
        std::cout << "./bin b m n k" << std::endl;
        std::cout << "./bin m n k" << std::endl;
        return -1;
    }

    tpm::Graph g{};
    tpm::PerfEngine pe{};
    tpm::Tensor *i0, *w0;
    if (transA)
        i0 = g.tensor({b, k, m});
    else
        i0 = g.tensor({b, m, k});
    if (transB)
        w0 = g.tensor({b, n, k});
    else
        w0 = g.tensor({b, k, n});
    auto gemm = g.matmul(i0, w0, transA, transB);
    auto i1 = gemm->getOutputs()[0];
    auto outDim = i1->getDims();
    std::cout << "Matmul: input = [ " << b << ", " << m << ", " << k
              << "], weight = [" << b << ", " << k << "," << n << "], "
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
