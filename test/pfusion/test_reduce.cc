#include "core/blob.h"
#include "core/graph.h"
#include "core/runtime.h"
#include "operators/element_wise.h"
#include "operators/extend.h"
#include "operators/gather.h"
#include "operators/matmul.h"
#include "operators/reduce_mean.h"
#include "operators/transpose.h"
#include "operators/unary.h"

#include "pfusion/memory_codegen.h"
#include "test.h"

namespace infini {

TEST(Graph, reduce_mean) {
    Runtime runtime = CpuRuntimeObj::getInstance();
    Graph g = make_ref<GraphObj>(runtime);
    Tensor t0 = g->addTensor({1, 128, 512}, DataType::Float32);
    Tensor t1 = g->addTensor({1, 128, 1}, DataType::Float32);
    g->dataMalloc();
    g->addOpWithOutputs<ReduceMeanObj>(t0, t1, Shape({2}));

    MemoryCodegen codegen;
    codegen.exportGraph(g, "reduce_mean.cu");
}

} // namespace infini
