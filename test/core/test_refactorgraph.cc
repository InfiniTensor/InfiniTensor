#include "core/blob.h"
#include "core/graph.h"
#include "core/runtime.h"
#include "operators/element_wise.h"
#include "operators/matmul.h"
#include "operators/reduce.h"
#include "operators/transpose.h"
#include "operators/unary.h"
#include "test.h"

namespace infini {

TEST(Graph, testMatmul) {
    Runtime runtime = NativeCpuRuntimeObj::getInstance();
    Graph g = make_ref<GraphObj>(runtime);
    Tensor tensor0 = g->addTensor({1, 3, 3, 5}, DataType::Float32);
    Tensor tensor1 = g->addTensor({1, 3, 5, 3}, DataType::Float32);
    Tensor tensor2 = g->addTensor({1, 3, 5, 3}, DataType::Float32);
    Tensor tensor3 = g->addTensor({1, 3, 3, 5}, DataType::Float32);
    Tensor tensor4 = g->addTensor({1, 3, 5, 5}, DataType::Float32);
    g->addOpWithOutputs<TransposeObj>(tensor0, tensor2, Shape({0, 1, 3, 2}));
    g->addOpWithOutputs<TransposeObj>(tensor1, tensor3, Shape({0, 1, 3, 2}));
    g->addOpWithOutputs<MatmulObj>(tensor2, tensor3, tensor4);
    g->optimize();
}

TEST(Graph, testLayerNorm) {
    Runtime runtime = NativeCpuRuntimeObj::getInstance();
    Graph g = make_ref<GraphObj>(runtime);
    auto tensor0 = g->addTensor({64, 101, 768}, DataType::Float32);
    auto tensor1 = g->addTensor({1, 101, 768}, DataType::Float32);
    auto tensor2 = g->addTensor({64, 101, 768}, DataType::Float32);
    auto tensor3 = g->addTensor({64, 101, 1}, DataType::Float32);
    auto tensor4 = g->addTensor({64, 101, 768}, DataType::Float32);
    auto tensor5 = g->addTensor({64, 101, 768}, DataType::Float32);
    auto tensor6 = g->addTensor({64, 101, 1}, DataType::Float32);
    auto tensor7 = g->addTensor({}, DataType::Float32);
    auto tensor8 = g->addTensor({64, 101, 1}, DataType::Float32);
    auto tensor9 = g->addTensor({64, 101, 1}, DataType::Float32);
    auto tensor10 = g->addTensor({64, 101, 768}, DataType::Float32);
    auto tensor11 = g->addTensor({768}, DataType::Float32);
    auto tensor12 = g->addTensor({64, 101, 768}, DataType::Float32);
    auto tensor13 = g->addTensor({768}, DataType::Float32);
    auto tensor14 = g->addTensor({64, 101, 768}, DataType::Float32);
    auto tensor15 = g->addTensor({}, DataType::Float32);
    auto tensor16 = g->addTensor({64, 101, 768}, DataType::Float32);

    g->addOpWithOutputs<AddObj>(tensor0, tensor1, tensor2);
    g->addOpWithOutputs<ReduceMeanObj>(tensor2, tensor3, Shape({2}), true);
    g->addOpWithOutputs<SubObj>(tensor3, tensor2, tensor4);
    g->addOpWithOutputs<PowObj>(tensor4, tensor15, tensor5);
    g->addOpWithOutputs<ReduceMeanObj>(tensor5, tensor6, Shape({2}), true);
    g->addOpWithOutputs<AddObj>(tensor6, tensor7, tensor8);
    g->addOpWithOutputs<SqrtObj>(tensor8, tensor9);
    g->addOpWithOutputs<DivObj>(tensor9, tensor4, tensor10);
    g->addOpWithOutputs<MulObj>(tensor10, tensor11, tensor12);
    g->addOpWithOutputs<AddObj>(tensor12, tensor13, tensor14);
    g->addOpWithOutputs<AddObj>(tensor14, tensor2, tensor16);
    g->optimize();
    // g->dataMalloc();
    // runtime->run(g);
}
} // namespace infini
