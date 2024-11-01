#include "code_gen/graph.h"
#include "code_gen/operator.h"
#include "test.h"
using namespace tpm;

TEST(ELIMINATOR_TEST_1, Cuda_codeGenerate) {
    auto g = Graph{};
    auto t0 = g.tensor({2, 6, 2, 2});
    auto op0 = dynamic_cast<SplitOp *>(g.split(t0, 1, {1, 2}));

    t0->dataRand();
    op0->computeShapeV();

    auto t1 = op0->getOutputs()[0];
    auto t2 = op0->getOutputs()[1];

    std::cout << "t1 size: " << dimToString(t1->getDims()) << std::endl;
    std::cout << "t2 size: " << dimToString(t2->getDims()) << std::endl;

    op0->computeV();
    t0->print();
    t1->print();
    t2->print();
}