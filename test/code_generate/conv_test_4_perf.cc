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
    int n = 64, c = 256, h = 14, w = 14, f = 256, wc = 256, r = 3, s = 3;
    int dh = 1, dw = 1;
    int sh = 1, sw = 1;
    int ph = 1, pw = 1;
    if (argc >= 9) {
        n = atoi(argv[1]);
        c = atoi(argv[2]);
        h = atoi(argv[3]);
        w = atoi(argv[4]);
        f = atoi(argv[5]);
        wc = atoi(argv[6]);
        r = atoi(argv[7]);
        s = atoi(argv[8]);
    } else if (argc == 8) {
        n = atoi(argv[1]);
        c = atoi(argv[2]);
        h = atoi(argv[3]);
        w = atoi(argv[4]);
        f = atoi(argv[5]);
        r = atoi(argv[6]);
        s = atoi(argv[7]);
    } else if (argc == 5) {
        n = atoi(argv[1]);
        c = atoi(argv[2]);
        h = atoi(argv[3]);
        w = h;
        f = atoi(argv[4]);
        wc = c;
    } else if (argc > 1 && argc != 11 && argc != 13 && argc != 15) {
        std::cout << "Arg formats:" << std::endl;
        std::cout << "./bin n c h w f wc r s [ph pw [sh sh [dh dw]]]"
                  << std::endl;
        std::cout << "./bin n c h w f r s" << std::endl;
        std::cout << "./bin n c insize f" << std::endl;
        return -1;
    }
    if (argc >= 11) {
        ph = atoi(argv[9]);
        pw = atoi(argv[10]);
    }
    if (argc >= 13) {
        sh = atoi(argv[11]);
        sw = atoi(argv[12]);
    }
    if (argc == 15) {
        dh = atoi(argv[13]);
        dw = atoi(argv[14]);
    }

    tpm::Graph g{};
    tpm::PerfEngine pe{};
    auto i0 = g.tensor({n, c, h, w});
    auto w0 = g.tensor({f, c, r, s});
    auto conv =
        dynamic_cast<tpm::ConvOp *>(g.conv(i0, w0, ph, pw, sh, sw, dh, dw));
    auto i1 = conv->getOutputs()[0];
    auto outDim = i1->getDims();
    conv->setAct(tpm::Operator::ActType::Relu);
    std::cout << "Conv: input = [ " << n << ", " << c << ", " << h << ", " << w
              << "], weight = [" << f << ", " << wc << ", " << r << ", " << s
              << "], "
              << "p = [" << ph << ", " << pw << "], "
              << "s = [" << sh << ", " << sw << "], "
              << "d = [" << dh << ", " << dw
              << "], output = " << tpm::dimToString(outDim) << std::endl;

    std::cout << conv->perf(&pe, 10, 2) << std::endl;

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
