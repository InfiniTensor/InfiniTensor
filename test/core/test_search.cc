#include "core/blob.h"
#include "core/graph.h"
#include "core/runtime.h"
#include "core/search_engine.h"
#include "nnet/nmutator.h"
#include "operators/matmul.h"
#include "operators/unary.h"
#include "test.h"

namespace infini {

TEST(Graph, search) {
    Runtime runtime = CpuRuntimeObj::getInstance();
    Graph g = make_ref<GraphObj>(runtime);
    Tensor i0 = g->addTensor({1, 2, 3}, DataType::UInt32);
    Tensor w0 = g->addTensor({1, 3, 4}, DataType::UInt32);
    Tensor o0 = g->addTensor({1, 2, 4}, DataType::UInt32);
    g->dataMalloc();
    i0->copyData(vector<uint32_t>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});
    w0->copyData(vector<uint32_t>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});
    auto matmul = g->addOpWithOutputs<MatmulObj>(i0, w0, o0);
    g->print();
    // check inputOf and outputsOf for tensor
    SearchEngine searchEngine(runtime, make_ref<NMutator>());
    searchEngine.run(g);
    // check execution results
}

} // namespace infini
