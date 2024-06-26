#include "core/graph.h"
#include "core/runtime.h"
#include "operators/conv.h"

#include "test.h"

namespace infini {

TEST(Conv3d, NativeCPU) {
    Runtime runtime = NativeCpuRuntimeObj::getInstance();
    Graph g = make_ref<GraphObj>(runtime);
    Tensor i0 = g->addTensor({1, 3, 4, 4, 4}, DataType::UInt32);
    Tensor w0 = g->addTensor({2, 3, 3, 3, 3}, DataType::UInt32);
    auto conv3d =
        g->addOp<Conv3dObj>(i0, w0, nullptr, 1, 1, 1, 1, 2, 1, 1, 1, 2);
    g->dataMalloc();
    i0->setData(IncrementalGenerator());
    w0->setData(IncrementalGenerator());
    runtime->run(g, true, true);
    // check answer
    auto ans =
        make_ref<TensorObj>(Shape{1, 2, 4, 2, 2}, DataType::UInt32, runtime);
    ans->dataMalloc();
    ans->copyin(vector<uint32_t>{
        113412, 110496, 175914, 171378, 172062, 167562, 264681, 257688,
        196254, 191178, 299673, 291816, 125796, 122328, 190818, 185454,
        261156, 256296, 415026, 407574, 417006, 409590, 658341, 646974,
        487854, 479862, 763317, 751086, 335748, 330336, 523242, 514962});
    auto o0 = conv3d->getOutput();
    EXPECT_TRUE(o0->equalData(ans));
}

} // namespace infini
