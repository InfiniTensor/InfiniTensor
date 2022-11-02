#include "core/blob.h"
#include "core/graph.h"
#include "core/runtime.h"
#include "operators/matmul.h"
#include "operators/transpose.h"
#include "operators/unary.h"
#include "pfusion/memory_codegen.h"
#include "test.h"

namespace infini {

TEST(Graph, build_and_run) {
    Runtime runtime = CpuRuntimeObj::getInstance();
    Graph g = make_ref<GraphObj>(runtime);
    Tensor t0 = g->addTensor({1024, 32, 32}, DataType::Float32);
    Tensor t1 = g->addTensor({1024, 32, 32}, DataType::Float32);
    Tensor t2 = g->addTensor({1024, 32, 32}, DataType::Float32);
    g->dataMalloc();
    g->addOpWithOutputs<AbsObj>(t0, t1);
    g->addOpWithOutputs<ReluObj>(t0, t1);
    MemoryCodegen codegen;
    codegen.exportGraph(g, "test.cu");
}

TEST(Graph, transpose) {
    Runtime runtime = CpuRuntimeObj::getInstance();
    Graph g = make_ref<GraphObj>(runtime);
    Tensor t0 = g->addTensor({32, 31, 33, 32}, DataType::Float32);
    Tensor t1 = g->addTensor({33, 32, 32, 31}, DataType::Float32);
    Tensor t2 = g->addTensor({33, 32, 32, 31}, DataType::Float32);
    g->dataMalloc();
    g->addOpWithOutputs<TransposeObj>(t0, t1, Shape{2, 0, 3, 1});
    g->addOpWithOutputs<ReluObj>(t1, t2);
    MemoryCodegen codegen;
    codegen.exportGraph(g, "transpose.cu");
}

} // namespace infini
