#include "core/blob.h"
#include "core/graph.h"
#include "core/runtime.h"
#include "operators/matmul.h"
#include "test.h"

namespace infini {

TEST(Graph, build_and_run) {
    Runtime runtime = CpuRuntimeObj::getInstance();
    Graph g = make_ref<GraphObj>(runtime);
    Tensor i0 = g->addTensor({1, 3, 4}, DataType::Float32);
    Tensor w0 = g->addTensor({1, 3, 4}, DataType::Float32);
    g->dataMalloc();
    i0->copyData(vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});
    w0->copyData(vector<float>{1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1});
    i0->save("i0.pb");
    w0->load("i0.pb");
}

} // namespace infini
