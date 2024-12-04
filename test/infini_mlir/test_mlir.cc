#include "core/graph.h"
#include "core/runtime.h"
#include "operators/element_wise.h"
#include "gtest/gtest.h"
#include "test.h"

namespace infini {
namespace infinimlir {

TEST(Graph, coverttomlir) {
    Runtime runtime = NativeCpuRuntimeObj::getInstance();
    Graph g = make_ref<GraphObj>(runtime);
    Tensor i1 = g->addTensor({2, 3, 4, 5}, DataType::UInt32);
    Tensor i2 = g->addTensor({2, 3, 4, 5}, DataType::UInt32);
    Tensor o = g->addTensor({2, 3, 4, 5}, DataType::UInt32);
    i1->setInput();
    i2->setWeight();
    o->setOutput();
    g->addOpWithOutputs<AddObj>(i1, i2, o);

    g->dataMalloc();
    i2->setData(OneGenerator());

    g->print();
    g->optimize();
    g->print();
}

} // namespace infinimlir
} // namespace infini
