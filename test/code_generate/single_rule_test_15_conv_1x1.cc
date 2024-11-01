#include "code_gen/generator.h"
#include "code_gen/graph.h"
#include "code_gen/operator.h"
#include "code_gen/search_engine.h"
#include "code_gen/tensor.h"
#include <cstdlib>
#include <iostream>
#include "test.h"
const int c = 256, h = 28, w = 28;
const int f = 128, r = 1, s = 1;

TEST(SINGLE_RULE_TEST_15_CONV_1x1, Cuda_codeGenerate) {
    // {
    //     int n = 1;
    //     auto g = new tpm::Graph();
    //     auto i0 = g->tensor({n, c, h, w});
    //     auto w0 = g->tensor({f, c, r, s});
    //     auto o1 = g->tensor({n, f, h, w});
    //     auto op0 = g->conv(i0, w0, o1, tpm::ConvOp::Same);

    //     auto sg = new tpm::SubGraph({op0});
    //     printf("%s:%d %lu\n", __FILE__, __LINE__, sg->getTensors().size());
    //     for (auto tensor : sg->getTensors())
    //         tensor->dataMalloc();
    //     for (auto tensor : sg->getInputs())
    //         tensor->dataRand();
    //     // for (auto op : sg->getOperators())
    //     //     op->compute();

    //     std::vector<std::shared_ptr<tpm::Operator>> all_ops;
    //     // all_ops.emplace_back(new tpm::TransposeOp(0, {0, 1, {-1, 2}, 3},
    //     2));
    //     // all_ops.emplace_back(new tpm::TransposeOp(2, {{0, 2}, 1, -1, 3},
    //     // -2)); all_ops.emplace_back(new tpm::ConvOp(tpm::ConvOp::Same, 1,
    //     1,
    //     // 1, 1));
    //     tpm::Generator mutant{};
    //     std::vector<tpm::SubGraph *> candidates;
    //     mutant.run(sg, candidates);
    //     std::cout << "candidates found: " << candidates.size() << std::endl;
    //     for (auto candidate : candidates)
    //         candidate->print();
    //     assert(candidates.size() == 1);
    //     delete g;
    //     delete sg;
    // }
    {
        int n = 16;
        auto g = new tpm::Graph();
        auto i0 = g->tensor({16, c, h, w});
        auto w0 = g->tensor({f, c, r, s});
        auto o1 = g->tensor({n, f, h, w});
        auto op0 = g->conv(i0, w0, o1, tpm::ConvOp::Same);

        auto sg = new tpm::SubGraph({op0});
        printf("%s:%d %lu\n", __FILE__, __LINE__, sg->getTensors().size());
        for (auto tensor : sg->getTensors())
            tensor->dataMalloc();
        for (auto tensor : sg->getInputs())
            tensor->dataRand();
        // for (auto op : sg->getOperators())
        //     op->compute();

        std::vector<std::shared_ptr<tpm::Operator>> all_ops;
        // all_ops.emplace_back(new tpm::TransposeOp(0, {0, 1, {-1, 2}, 3}, 2));
        // all_ops.emplace_back(new tpm::TransposeOp(2, {{0, 2}, 1, -1, 3},
        // -2)); all_ops.emplace_back(new tpm::ConvOp(tpm::ConvOp::Same, 1, 1,
        // 1, 1));
        tpm::Generator mutant{};
        std::vector<tpm::SubGraph *> candidates;
        mutant.run(sg, candidates);
        std::cout << "candidates found: " << candidates.size() << std::endl;
        for (auto candidate : candidates)
            candidate->print();
        assert(candidates.size() == 1);
        delete g;
        delete sg;
    }

    // assert(candidates.size() == 2);

    // auto g1 = new tpm::Graph();
    // auto t0 = g1->tensor({n, c, h, w});
    // auto t1 = g1->tensor({f, c, r, s});
    // auto p0 = g1->conv(t0, t1, 1, 1, 1, 1, 1, 1);
    // t0->dataMalloc();
    // t1->dataMalloc();
    // auto t2 = p0->compute();
    // std::cout << "t0: " << std::endl;
    // t0->print();
    // std::cout << "t1: " << std::endl;
    // t1->print();
    // std::cout << "t2: " << std::endl;
    // t2->print();
    // auto p1 = g1->transpose(t0, 0, {0, 1, {-1, 2}, 3}, 2);
    // auto t3 = p1->compute();
    // std::cout << "t3: " << std::endl;
    // t3->print();
    // auto p2 = g1->conv(t3, t1, 1, 1, 1, 1, 1, 1);
    // auto t4 = p2->compute();
    // std::cout << "t4: " << std::endl;
    // t4->print();
    // auto p3 = g1->transpose(t4, 2, {{0, 2}, 1, -1, 3}, -2);
    // auto t5 = p3->compute();
    // std::cout << "t5: " << std::endl;
    // t5->print();

}
