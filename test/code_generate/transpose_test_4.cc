#include "code_gen/generator.h"
#include "code_gen/graph.h"
#include "code_gen/operator.h"
#include "code_gen/search_engine.h"
#include "code_gen/tensor.h"
#include <iostream>
#include "test.h"

const int n = 2, c = 1, h = 8, w = 8;

TEST(TRNASPOSE_TEST_4, Cuda_codeGenerate) {
    auto i0 = new tpm::Tensor({n, c, h, w});
    // tpm::Perm perm = {0, 1, {-1, 2}, 3};
    // int split = 2, factor = 2;
    tpm::Perm perm = {0, 1, {-1, 2}, 3};
    int split = 0, factor = 2;
    auto trans1 = new tpm::TransposeOp(i0, split, perm, factor);
    auto trans2 = new tpm::TransposeOp(i0, split, perm, factor);
    auto o1 = trans1->getOutputs()[0];
    auto o2 = trans2->getOutputs()[0];
    i0->dataMalloc();
    o1->dataMalloc();
    o2->dataMalloc();
    auto i0d = i0->getDataPtr();
    for (size_t i = 0; i < i0->size(); ++i)
        i0d[i] = i;

    trans1->compute();

    o2->itInit();
    while (o2->itValid()) {
        auto it = o2->itGet();
        std::vector<tpm::DimRange> drInputs;
        std::function<bool()> run;
        std::tie(drInputs, run) = trans2->compute(tpm::DimRange(it));
        run();
        assert(drInputs.size() == 1);
        auto itInput = drInputs[0].getBegin();
        std::cout << "output: " << tpm::dimToString(it)
                  << ", input: " << tpm::dimToString(itInput) << std::endl;
        o2->itNext();
    }

    auto o1d = o1->getDataPtr();
    auto o2d = o2->getDataPtr();
    int total = 0, equal = 0;
    for (size_t i = 0; i < o1->size(); ++i) {
        total++;
        if (o1d[i] == o2d[i])
            equal++;
    }
    std::cout << "equal/total = " << equal << "/" << total << " = "
              << (double)equal / total << std::endl;
    assert(equal == total);

    delete trans1;
    delete trans2;
    delete i0;
    delete o1;
    delete o2;
}
