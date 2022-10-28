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

TEST(Graph, bert_0) {
    Runtime runtime = CpuRuntimeObj::getInstance();
    Graph g = make_ref<GraphObj>(runtime);
    Tensor t0 = g->addTensor({1, 128}, DataType::Float32);
    Tensor t1 = g->addTensor({30522, 512}, DataType::Float32);
    Tensor t2 = g->addTensor({1, 128, 512}, DataType::Float32);

    Tensor t3 = g->addTensor({1, 128, 512}, DataType::Float32);
    Tensor t4 = g->addTensor({1, 128, 512}, DataType::Float32);

    Tensor t5 = g->addTensor({1, 128, 512}, DataType::Float32);
    Tensor t6 = g->addTensor({1, 128, 512}, DataType::Float32);

    Tensor t7 = g->addTensor({1, 128, 1}, DataType::Float32);

    Tensor t8 = g->addTensor({1, 128, 512}, DataType::Float32);

    Tensor t9 = g->addTensor({1, 128, 512}, DataType::Float32);
    Tensor t10 = g->addTensor({1, 128, 512}, DataType::Float32);

    g->dataMalloc();
    g->addOpWithOutputs<GatherObj>(t1, t0, t2, 0);
    g->addOpWithOutputs<AddObj>(t2, t3, t4);
    g->addOpWithOutputs<AddObj>(t4, t5, t6);
    g->addOpWithOutputs<ReduceMeanObj>(t6, t7, Shape({2}));
    g->addOpWithOutputs<ExtendObj>(t7, t8, 2, 511);
    g->addOpWithOutputs<SubObj>(t8, t9, t10);

    MemoryCodegen codegen;
    codegen.exportCode(g, "bert_0.cu");
}

} // namespace infini
