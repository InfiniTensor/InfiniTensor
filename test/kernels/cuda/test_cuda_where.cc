#include "core/graph.h"
#include "core/runtime.h"
#include "cuda/cuda_runtime.h"
#include "cuda/cuda_utility.h"
#include "operators/where.h"

#include "test.h"

namespace infini {

TEST(Where, Cuda) {
    Runtime runtime = NativeCpuRuntimeObj::getInstance();
    Graph gCpu = make_ref<GraphObj>(runtime);

    auto condition = gCpu->addTensor({2, 2, 3, 1}, DataType::Int32);
    auto inputx = gCpu->addTensor({2, 2, 3, 1}, DataType::Float32);
    auto inputy = gCpu->addTensor({2, 2, 3, 1}, DataType::Float32);
    gCpu->dataMalloc();
    condition->copyin(vector<int>{0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 1, 1});//使用copyin可以自动填充数据，自动reshape
    inputx->setData(IncrementalGenerator());//数据为0到size-1的数
    inputy->setData(ZeroGenerator());//元素全是0

    auto cudaRuntime = make_ref<CudaRuntimeObj>();
    Graph gCuda = make_ref<GraphObj>(cudaRuntime);

    auto conditionGpu = gCuda->cloneTensor(condition);
    auto inputxGpu = gCuda->cloneTensor(inputx);
    auto inputyGpu = gCuda->cloneTensor(inputy);

    auto op =
        gCuda->addOp<WhereObj>(inputxGpu, inputyGpu, conditionGpu, nullptr);//参数的输入参考WhereObj类
    gCuda->dataMalloc();
    conditionGpu->copyin(vector<int>{0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 1, 1});
    inputxGpu->setData(IncrementalGenerator());
    inputyGpu->setData(ZeroGenerator());
    cudaRuntime->run(gCuda);//这里才是正式跑cuda代码

    // cudaPrintTensor(op->getOutput());
    //  copy output from CUDA to CPU
    auto oCpu = gCpu->cloneTensor(op->getOutput());//把cuda上的结果传回CPU
    oCpu->printData();//->printData打印结果，方便检查代码的正确性
    EXPECT_TRUE(
        oCpu->equalData(vector<float>{0.,  1.,  2.,  0.,  0.,  0.,  6.,  7.,  0.,  9., 10., 11.}));
}//这个结果是通过python计算出来的

} // namespace infini
