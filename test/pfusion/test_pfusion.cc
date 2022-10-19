#include "core/blob.h"
#include "core/graph.h"
#include "core/runtime.h"
#include "operators/matmul.h"
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
    codegen.export_code(g, "test.cu");
}

} // namespace infini
