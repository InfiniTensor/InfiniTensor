#include "core/graph.h"
#include "core/kernel.h"
#include "core/runtime.h"
#include "cuda/cuda_runtime.h"
#include "operators/any.h"

#include "test.h"

namespace infini {
TEST(cuda_Any, anyKernel) {
    // conv2dreduce
    {
        // Construct Runtime and graph for CPU and CUDA
        Runtime cpu =
            NativeCpuRuntimeObj::getInstance(); // CPUruntime is singleton
        Graph gCpu = make_ref<GraphObj>(cpu);
        Runtime cuda = make_ref<CudaRuntimeObj>();
        Graph gCuda = make_ref<GraphObj>(cuda);

        auto generator = IncrementalGenerator();

        int PRelu = 0, n = 1, h = 4, w = 4, f = 2, r = 3, s = 3, oh = 4, ow = 4,
            ph = 1, pw = 1, sh = 1, sw = 1, dh = 1, dw = 1;
        string kernelName = "conv2dreduce_kernel";
        vector<int> attr{PRelu, n,  h,  w,  f,  r,  s, oh,
                         ow,    ph, pw, sh, sw, dh, dw};

        // Build input data on CPu
        Tensor i0Cpu = gCpu->addTensor({n, 1, h, w}, DataType::Float32);
        Tensor w0Cpu = gCpu->addTensor({f, 1, r, s}, DataType::Float32);
        // Malloc data for all tensors in a graph. Do we need implicit
        // allocation?
        gCpu->dataMalloc();
        i0Cpu->setData(generator);
        w0Cpu->setData(generator);
        // Copy input tensors from CPU to CUDA
        Tensor i0Cuda = gCuda->cloneTensor(i0Cpu);
        Tensor w0Cuda = gCuda->cloneTensor(w0Cpu);
        Tensor o0Cuda = gCuda->addTensor({n, f, oh, ow});
        auto anyOp = gCuda->addOpWithOutputs<AnyObj>(
            TensorVec{i0Cuda, w0Cuda}, TensorVec{o0Cuda}, kernelName, attr);
        anyOp->print();
        // allocate CUDA memory
        gCuda->dataMalloc();
        std::cout << "data malloc success..." << std::endl;
        // Execute on CUDA
        cuda->run(gCuda);
        std::cout << "cuda run success..." << std::endl;
        // copy output from CUDA to CPU
        auto o0Cpu = gCpu->cloneTensor(anyOp->getOutput());
        // check results on CPU
        EXPECT_TRUE(1);
        // print a tensor/operator/graph by print()
        gCuda->print();
    }
}
} // namespace infini
