#include "core/blob.h"
#include "core/graph.h"
#include "core/runtime.h"
#include "operators/element_wise.h"
#include "operators/matmul.h"
#include "operators/transpose.h"
#include "operators/unary.h"
#include "pfusion/memory_codegen.h"
#include "test.h"

namespace infini {

TEST(Graph, SAR_DRN_0) {
    Runtime runtime = CpuRuntimeObj::getInstance();
    Graph g = make_ref<GraphObj>(runtime);
    Tensor t0 = g->addTensor({1, 64, 512, 512}, DataType::Float32);
    Tensor t1 = g->addTensor({1, 64, 512, 512}, DataType::Float32);
    Tensor t2 = g->addTensor({1, 64, 512, 512}, DataType::Float32);
    Tensor t3 = g->addTensor({1, 64, 512, 512}, DataType::Float32);
    g->dataMalloc();
    g->addOpWithOutputs<ReluObj>(t0, t1);
    g->addOpWithOutputs<AddObj>(t1, t2, t3);
    MemoryCodegen codegen;
    codegen.exportGraph(g, "sar_drn_0.cu");
}

TEST(Graph, SAR_DRN_1) {
    Runtime runtime = CpuRuntimeObj::getInstance();
    Graph g = make_ref<GraphObj>(runtime);
    Tensor t0 = g->addTensor({1, 1, 512, 512}, DataType::Float32);
    Tensor t1 = g->addTensor({1, 1, 512, 512}, DataType::Float32);
    Tensor t2 = g->addTensor({1, 1, 512, 512}, DataType::Float32);
    Tensor t3 = g->addTensor({1, 1, 512, 512}, DataType::Float32);
    g->dataMalloc();
    g->addOpWithOutputs<ReluObj>(t0, t1);
    g->addOpWithOutputs<SubObj>(t1, t2, t3);
    MemoryCodegen codegen;
    codegen.exportGraph(g, "sar_drn_1.cu");
}

} // namespace infini
