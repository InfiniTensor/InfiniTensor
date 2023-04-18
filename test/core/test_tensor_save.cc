#include "core/blob.h"
#include "core/graph.h"
#include "core/runtime.h"
#include "operators/matmul.h"
#include "test.h"

namespace infini {

TEST(Prtotbuf, save_and_load) {
#ifdef TENSOR_PROTOBUF
    Runtime runtime = NativeCpuRuntimeObj::getInstance();
    Graph g = make_ref<GraphObj>(runtime);
    Tensor i0 = g->addTensor({1, 3, 4}, DataType::Float32);
    Tensor w0 = g->addTensor({1, 3, 4}, DataType::Float32);
    Tensor u0 = g->addTensor({1, 3, 4}, DataType::UInt32);
    Tensor u1 = g->addTensor({1, 3, 4}, DataType::UInt32);
    g->dataMalloc();
    i0->copyin(vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});
    w0->copyin(vector<float>{1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1});
    u0->copyin(vector<uint32_t>{1, 3, 5, 7, 9, 2, 4, 6, 8, 10, 0, 0});
    u1->copyin(vector<uint32_t>{1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0});
    i0->save("i0.pb");
    w0->printData();
    w0->load("i0.pb");
    w0->printData();
    EXPECT_TRUE(w0->equalData(i0));
    u0->save("u.pb");
    u1->printData();
    u1->load("u.pb");
    u1->printData();
    EXPECT_TRUE(u1->equalData(u0));
#endif
}

} // namespace infini
