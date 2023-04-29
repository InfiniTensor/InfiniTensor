#include "core/graph.h"
#include "core/kernel.h"
#include "core/runtime.h"
#include "cuda/cuda_runtime.h"
#include "cuda_profiler_api.h"
#include "operators/G2BMM.h"

#include "test.h"

namespace infini {

constexpr int bs = 1, seqlen = 10000, w = 1000, featlen = 512, heads = 8, d = 4,
              hidden = featlen, hiddenPerHead = hidden / heads;

void testOpG2BMMOrigin(bool printGraph = false) {
    auto generator = IncrementalGenerator();
    // Construct Runtime and graph for CPU and CUDA
    Runtime cpu = NativeCpuRuntimeObj::getInstance(); // CPUruntime is singleton
    Graph gCpu = make_ref<GraphObj>(cpu);
    Runtime cuda = make_ref<CudaRuntimeObj>();
    Graph gCuda = make_ref<GraphObj>(cuda);
    // Set input data on CPU in a CPU Graph
    Tensor qCpu = gCpu->addTensor({bs * heads, seqlen, hiddenPerHead},
                                  DataType::Float32, TensorType::Other);
    Tensor kCpu = gCpu->addTensor({bs * heads, seqlen, hiddenPerHead},
                                  DataType::Float32, TensorType::Other);
    // Malloc data for all tensors in a graph. Do we need implicit
    // allocation?
    gCpu->dataMalloc();
    qCpu->setData(generator);
    kCpu->setData(generator);

    // Copy input tensors from CPU to CUDA
    Tensor qCuda = gCuda->cloneTensor(qCpu);
    Tensor kCuda = gCuda->cloneTensor(kCpu);
    // Build CUDA graph
    gCuda->addOp<G2BMMObj>(qCuda, kCuda, nullptr, w, d);
    // allocate CUDA memory
    gCuda->dataMalloc();
    // Execute on CUDA
    cuda->run(gCuda, true);
    std::cout << "Time: " << cuda->getPerfTime(gCuda) << " ms" << std::endl;
    cudaProfilerStart();
    cuda->run(gCuda);
    cudaProfilerStop();

    if (printGraph) {
        // print a tensor/operator/graph by print()
        gCuda->print();
    }
}

void testOpG2BMMOptimized(bool printGraph = false) {
    auto generator = IncrementalGenerator();
    // Construct Runtime and graph for CPU and CUDA
    Runtime cpu = NativeCpuRuntimeObj::getInstance(); // CPUruntime is singleton
    Graph gCpu = make_ref<GraphObj>(cpu);
    Runtime cuda = make_ref<CudaRuntimeObj>();
    Graph gCuda = make_ref<GraphObj>(cuda);
    // Set input data on CPU in a CPU Graph
    Tensor qCpu = gCpu->addTensor({bs * heads, seqlen, hiddenPerHead},
                                  DataType::Float32, TensorType::Other);
    Tensor kCpu = gCpu->addTensor({bs * heads, seqlen, hiddenPerHead},
                                  DataType::Float32, TensorType::Other);
    // Malloc data for all tensors in a graph. Do we need implicit
    // allocation?
    gCpu->dataMalloc();
    qCpu->setData(generator);
    kCpu->setData(generator);

    // Copy input tensors from CPU to CUDA
    Tensor qCuda = gCuda->cloneTensor(qCpu);
    Tensor kCuda = gCuda->cloneTensor(kCpu);
    // Build CUDA graph
    gCuda->addOp<G2BMMObj>(qCuda, kCuda, nullptr, w, 1);
    // allocate CUDA memory
    gCuda->dataMalloc();
    // Execute on CUDA
    cuda->run(gCuda, true);
    std::cout << "Time: " << cuda->getPerfTime(gCuda) << " ms" << std::endl;
    cudaProfilerStart();
    cuda->run(gCuda);
    cudaProfilerStop();

    if (printGraph) {
        // print a tensor/operator/graph by print()
        gCuda->print();
    }
}

constexpr int rounds = 100;

TEST(op_G2BMM, origin) { testOpG2BMMOrigin(); }

TEST(op_G2BMM, optimized) { testOpG2BMMOptimized(); }

} // namespace infini
