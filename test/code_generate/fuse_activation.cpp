#include "code_gen/graph.h"
#include "code_gen/operator.h"
#include "code_gen/search_engine.h"
#include "code_gen/tensor.h"
#include "test.h"

TEST(FUSE_ACTIVATION, Cuda_codeGenerate) {
    // matmul -> transpose -> relu
    auto g = new tpm::Graph();
    auto i0 = g->tensor({1, 32, 32});
    auto w1 = g->tensor({1, 32, 32});
    auto i1 = g->tensor({1, 32, 32});
    auto i2 = g->tensor({1, 32, 32});
    auto i3 = g->tensor({1, 32, 32});
    g->matmul(i0, w1, i1);
    g->transpose(i1, i2, -1, {0, 2, 1});
    g->relu(i2, i3);

    g->updateConnection();
    tpm::SearchEngine engine(std::make_shared<tpm::Generator>());

    std::shared_ptr<tpm::SubGraph> h(new tpm::SubGraph(g->getOperators()));
    auto newG = engine.fuse(h);
    newG->print();
}
