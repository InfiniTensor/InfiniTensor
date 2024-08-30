#include "code_gen/graph.h"
#include "code_gen/operator.h"

using namespace tpm;

int main() {
    auto g = Graph{};
    auto t0 = g.tensor({2, 2, 3, 3});
    auto op0 = g.extend(t0, 1, 1);

    t0->dataRand();
    op0->computeShape();

    auto t1 = op0->getOutputs()[0];

    std::cout << "t1 size: " << dimToString(t1->getDims()) << std::endl;

    op0->compute();
    t0->print();
    t1->print();

    return 0;
}