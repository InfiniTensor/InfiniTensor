#include "code_gen/graph.h"
#include "code_gen/operator.h"
#include "test.h"

using namespace tpm;

TEST(SPLIT_TEST_2, Cuda_codeGenerate) {
    auto g = Graph{};
    auto t0 = g.tensor({2, 6, 2, 2});
    auto split0 = dynamic_cast<SplitOp *>(g.split(t0, 1, {1, 2}));
    auto split1 = dynamic_cast<SplitOp *>(g.split(t0, 1, {1, 2}));

    t0->dataRand();
    split0->computeShapeV();

    auto t1 = split0->getOutputs()[0];
    auto t2 = split0->getOutputs()[1];

    std::cout << "t1 size: " << dimToString(t1->getDims()) << std::endl;
    std::cout << "t2 size: " << dimToString(t2->getDims()) << std::endl;

    split0->computeV();
    t0->print();
    t1->print();
    t2->print();

    split1->computeShapeV();
    auto o1 = split1->getOutputs()[0];
    auto o2 = split1->getOutputs()[1];
    o1->dataMalloc();
    o2->dataMalloc();

    o1->itInit();
    while (o1->itValid()) {
        auto it = o1->itGet();
        std::vector<tpm::DimRange> drInputs;
        std::function<bool()> run;
        std::tie(drInputs, run) = split1->compute(0, tpm::DimRange(it));
        run();
        assert(drInputs.size() == 1);
        auto itInput = drInputs[0].getBegin();
        o1->itNext();
    }
    o1->setComputed();
    o2->itInit();
    while (o2->itValid()) {
        auto it = o2->itGet();
        std::vector<tpm::DimRange> drInputs;
        std::function<bool()> run;
        std::tie(drInputs, run) = split1->compute(1, tpm::DimRange(it));
        run();
        assert(drInputs.size() == 1);
        auto itInput = drInputs[0].getBegin();
        o2->itNext();
    }
    o2->setComputed();
    std::cout << "t1" << std::endl;
    t1->print();
    o1->print();
    std::cout << "t2" << std::endl;
    t2->print();
    o2->print();

    int total = 0, equal = 0;
    for (size_t i = 0; i < t1->size(); ++i) {
        total++;
        if (o1->getData(i) == t1->getData(i))
            equal++;
    }
    for (size_t i = 0; i < t2->size(); ++i) {
        total++;
        if (o2->getData(i) == t2->getData(i))
            equal++;
    }
    std::cout << "equal/total=" << equal << "/" << total << std::endl;
}