#include "code_gen/code_engine.h"
#include "code_gen/graph.h"
#include "code_gen/operator.h"
#include "code_gen/search_engine.h"
#include "code_gen/tensor.h"
#include "test.h"

const int n = 1, c = 1, h = 7, w = 7;
const int f = 1, wc = 1, r = 1, s = 3;
using namespace tpm;
TEST(CONV_TEST_5_TRANS_KERNEL, Cuda_codeGenerate) {
    // conv7x7->relu->conv3x3->relu->conv3x3->relu->conv3x3->relu->conv3x3->relu
    auto g1 = Graph{};
    auto i0 = g1.tensor({n, c, h, w});
    auto w0 = g1.tensor({f, wc, r, s});
    g1.conv(i0, w0, ConvOp::Same);
    g1.updateConnection();
    auto o1 = g1.getOutputs()[0];

    auto g2 = Graph{};
    auto i1 = g2.tensor({n, c, h, w});
    auto w1 = g2.tensor({f, wc, r, s});
    auto trans1 = g2.transpose(i1, -1, {0, 1, 3, 2});
    auto trans2 = g2.transpose(w1, -1, {0, 1, 3, 2});
    auto i2 = trans1->getOutputs()[0];
    auto w2 = trans2->getOutputs()[0];
    auto conv2 = g2.conv(i2, w2, ConvOp::Same);
    auto i3 = conv2->getOutputs()[0];
    auto trans3 = g2.transpose(i3, -1, {0, 1, 3, 2});
    auto i4 = trans3->getOutputs()[0];
    g2.updateConnection();
    auto o2 = g2.getOutputs()[0];

    for (auto tensor : g1.getTensors())
        tensor->dataMalloc();

    for (auto tensor : g2.getTensors())
        tensor->dataMalloc();

    auto i0ptr = i0->getDataPtr(), w0ptr = w0->getDataPtr();
    auto i1ptr = i1->getDataPtr(), w1ptr = w1->getDataPtr();
    i0->setComputed();
    i1->setComputed();
    w0->setComputed();
    w1->setComputed();
    for (size_t i = 0; i < i0->size(); ++i)
        i0ptr[i] = i;
    for (size_t i = 0; i < w0->size(); ++i)
        w0ptr[i] = i;
    // w0ptr[3] = 0;
    // w0ptr[4] = 1;
    // w0ptr[5] = 2;
    for (size_t i = 0; i < i1->size(); ++i)
        i1ptr[i] = i;
    for (size_t i = 0; i < w1->size(); ++i)
        w1ptr[i] = i;
    // w1ptr[3] = 0;
    // w1ptr[4] = 1;
    // w1ptr[5] = 2;

    for (auto op : g1.getOperators())
        op->compute();

    for (auto op : g2.getOperators())
        op->compute();

    std::cout << "i1:" << std::endl;
    i1->print();
    std::cout << "i2:" << std::endl;
    i2->print();
    std::cout << "w1:" << std::endl;
    w1->print();
    std::cout << "w2:" << std::endl;
    w2->print();
    std::cout << "i3:" << std::endl;
    i3->print();
    std::cout << "i4:" << std::endl;
    i4->print();

    std::cout << "o1:" << std::endl;
    o1->print();
    std::cout << "o2:" << std::endl;
    o2->print();
    int equal = 0, total = 0;
    auto o1ptr = o1->getDataPtr(), o2ptr = o2->getDataPtr();
    for (size_t i = 0, iEnd = o1->size(); i < iEnd; ++i) {
        if (o1ptr[i] == o2ptr[i])
            equal++;
        total++;
    }
    std::cout << "equal/total = " << equal << "/" << total << std::endl;
}
