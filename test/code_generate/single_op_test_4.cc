#include "code_gen/code_engine.h"
#include "code_gen/graph.h"
#include "code_gen/operator.h"
#include "code_gen/search_engine.h"
#include "code_gen/tensor.h"

int main(int argc, char **argv) {
    auto g = new tpm::Graph();
    int n = 64, c = 256, h = 14, w = 14, f = 256, wc = 256, r = 3, s = 3;
    if (argc == 9) {
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
    } else if (argc > 1) {
        std::cout << "Arg formats:" << std::endl;
        std::cout << "./bin n c h w f wc r s" << std::endl;
        std::cout << "./bin n c h w f r s" << std::endl;
        std::cout << "./bin n c insize f" << std::endl;
        return -1;
    }
    std::cout << "Conv: [ " << n << ", " << c << ", " << h << ", " << w
              << "], [" << f << ", " << wc << ", " << r << ", " << s << "]"
              << std::endl;
    auto i8 = g->tensor({n, c, h, w});
    auto i9 = g->tensor({n, f, h, w});

    auto w9 = g->tensor({f, wc, r, s});

    g->conv(i8, w9, i9, 1, 1, 1, 1);

    g->updateConnection();

    std::shared_ptr<tpm::SubGraph> graph, bestGraph;
    graph = std::make_shared<tpm::SubGraph>(g->getOperators());
    tpm::SearchEngine searchEngine(std::make_shared<tpm::Generator>());
    searchEngine.run(graph, bestGraph);
    tpm::CodeEngine codeEngine;
    auto perfEngine = searchEngine.exportPerfEngine();
    codeEngine.importPerfEngine(perfEngine);
    codeEngine.genCode(bestGraph, "res.cu");

    return 0;
}
