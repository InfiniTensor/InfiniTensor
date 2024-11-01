#include "code_gen/graph.h"
#include "code_gen/operator.h"
#include "test.h"

using namespace tpm;

TEST(CONCAT_TEST_1, Cuda_codeGenerate) {
    auto g = Graph{};
    auto t0 = g.tensor({2, 2, 2, 2});
    auto t1 = g.tensor({2, 3, 2, 2});
    auto op0 = g.concat({t0, t1}, 1);

    t0->dataRand();
    t1->dataRand();
    op0->computeShape();

    auto t2 = op0->getOutputs()[0];

    std::cout << "t2 size: " << dimToString(t2->getDims()) << std::endl;

    op0->compute();
    t0->print();
    t1->print();
    t2->print();
}