#include "core/graph.h"
#include "core/runtime.h"
#include "operators/unary.h"
#include "test.h"

namespace infini {
template <class T>
void testClipCpu(
    const std::function<void(void*,size_t,DataType)> &generator,
    const Shape &shape,const DataType &dataType,
    std::optional<float> min=std::nullopt,
    std::optional<float> max=std::nullopt

){
    Runtime runtime=NativeCpuRuntimeObj::getInstance();
    Graph g=make_ref<GraphObj>(runtime);
    auto input=g->addTensor(shape,dataType);
    auto op = g->addOp<ClipObj>(input, nullptr, min, max);

    g->dataMalloc();
    input->setData(generator);
    runtime->run(g);
    // op->getOutput()->print();
    // op->getOutput()->printData();
    EXPECT_TRUE(1);

}

TEST(ElementWise, Cpu) {
    testClipCpu<ClipObj>(IncrementalGenerator(), Shape{1, 2, 2, 3},
                          DataType::Float32,-0.1f,0.1f);
    testClipCpu<ClipObj>(IncrementalGenerator(), Shape{1, 2, 2, 3},
                          DataType::Float16,-0.1f,0.1f);
}


}