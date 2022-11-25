#include "core/blob.h"
#include "core/dummy_mutator.h"
#include "core/graph.h"
#include "core/runtime.h"
#include "core/search_engine.h"
#include "nnet/nmutator.h"
#include "operators/conv.h"
#include "operators/element_wise.h"
#include "operators/matmul.h"
#include "operators/unary.h"
#include "test.h"

namespace infini {

// TEST(Graph, search) {
//     Runtime runtime = NativeCpuRuntimeObj::getInstance();
//     Graph g = make_ref<GraphObj>(runtime);
//     Tensor i0 = g->addTensor({1, 2, 3}, DataType::UInt32);
//     Tensor w0 = g->addTensor({1, 3, 4}, DataType::UInt32);
//     Tensor o0 = g->addTensor({1, 2, 4}, DataType::UInt32);
//     g->dataMalloc();
//     i0->copyin(vector<uint32_t>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});
//     w0->copyin(vector<uint32_t>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});
//     auto matmul = g->addOpWithOutputs<MatmulObj>(i0, w0, o0);
//     g->print();
//     // check targets and source for tensor
//     SearchEngine searchEngine(runtime, make_ref<NMutator>());
//     searchEngine.run(g);
//     // check execution results
// }

TEST(Graph, search_withdm) {
    Runtime runtime = NativeCpuRuntimeObj::getInstance();
    Graph g = make_ref<GraphObj>(runtime);
    Tensor t0 = g->addTensor({1, 3, 224, 224});
    Tensor w0 = g->addTensor({3, 3, 3, 3});
    Tensor t1 = g->addTensor({1, 3, 224, 224});
    Tensor t2 = g->addTensor({1, 3, 224, 224});
    Tensor t3 = g->addTensor({1, 3, 224, 224});
    Tensor w3 = g->addTensor({3, 3, 3, 3});
    Tensor t4 = g->addTensor({1, 3, 224, 224});
    Tensor t5 = g->addTensor({1, 3, 224, 224});
    Tensor t6 = g->addTensor({1, 3, 224, 224});
    auto conv0 = g->addOpWithOutputs<ConvObj>(t0, w0, t1, 1, 1);
    auto add0 = g->addOpWithOutputs<AddObj>(t1, t2, t3);
    auto conv1 = g->addOpWithOutputs<ConvObj>(t3, w3, t4, 1, 1);
    auto add1 = g->addOpWithOutputs<AddObj>(t4, t5, t6);
    g->dataMalloc();
    // check targets and source for tensor
    SearchEngine searchEngine(runtime, make_ref<DummyMutator>(10));
    searchEngine.run(g);
    // check execution results
}

// TEST(DummyMutator, run) {
//     Runtime runtime = NativeCpuRuntimeObj::getInstance();
//     Graph g = make_ref<GraphObj>(runtime);
//     Tensor i0 = g->addTensor({1, 3, 224, 224});
//     Tensor w0 = g->addTensor({2, 3, 3, 3});
//     auto matmul = g->addOp<ConvObj>(i0, w0, nullptr, 1, 1);
//     DummyMutator m(10);
//     auto mutations = m.run(g);
//     g->print();
//     for (auto gg : mutations) {
//         gg->print();
//     }
// }

// TEST(DummyMutator, fuse) {
//     Runtime runtime = NativeCpuRuntimeObj::getInstance();
//     Graph g = make_ref<GraphObj>(runtime);
//     Tensor i0 = g->addTensor({1, 2, 3});
//     Tensor w0 = g->addTensor({1, 3, 4});
//     Tensor i1 = g->addTensor({1, 2, 3});
//     Tensor w1 = g->addTensor({1, 3, 4});
//     auto matmul0 = g->addOp<MatmulObj>(i0, w0, nullptr);
//     auto matmul1 = g->addOp<MatmulObj>(i1, w1, nullptr);
//     DummyMutator m(10);
//     auto mutations = m.mergeMultiBranch(g);
//     g->print();
//     for (auto gg : mutations) {
//         gg->print();
//     }
// }

} // namespace infini
