#include "operators/matmul.h"
#include "core/kernel.h"
#include "core/tensor.h"
#include "native/cpu/ops/gemm/gemm.h"
#ifdef WITH_NVIDIA
#include "native/cuda/nvidia/ops/gemm/cublas.h"
#include "native/cuda/nvidia/ops/gemm/cublaslt.h"
#endif
#ifdef WITH_CAMBRICON
#include "native/cambricon/ops/gemm/cnblas.h"
#endif
#ifdef WITH_ASCEND
#include "native/ascend/ops/gemm/kernel.h"
#endif
#ifdef WITH_ILUVATAR
#include "native/cuda/iluvatar/ops/gemm/cublas.h"
#endif
#ifdef WITH_METAX
#include "native/cuda/metax/ops/gemm/mcblas.h"
#endif
#ifdef WITH_MOORE
#include "native/cuda/moore/ops/gemm/mublas.h"
#endif
#ifdef WITH_TORCH
#include "torch/ops/gemm/gemm.h"
#endif

namespace infini {

class MatmulInfiniOpsKernel : public KernelWithoutConfig {
    void compute(const Operator &op, const RuntimeObj *context) const override {
        auto matmulOp = as<MatmulObj>(op);

        auto a = toInfiniOpsTensor(matmulOp->getInputs(0).get());
        auto b = toInfiniOpsTensor(matmulOp->getInputs(1).get());
        auto *out0 = matmulOp->getOutput().get();
        auto output = toInfiniOpsTensor(out0);

        infini::ops::Handle handle = context->makeHandle();
        infini::ops::Config config;
        config.set_implementation_index(
            context->resolveImplementationIndex<infini::ops::Gemm>());

        bool transA = matmulOp->getTransA();
        bool transB = matmulOp->getTransB();

        if (auto bias = matmulOp->getBias()) {
            // Fused matmul + bias (element-wise): Y = alpha * A @ B + beta * C
            // 1. Copy bias data to output buffer
            // 2. Call Gemm with beta=1 to accumulate: output = A @ B + output
            auto *biasObj = bias.get();
            context->copyBlob(out0, biasObj);

            infini::ops::Gemm::Call(handle, config, a, b,
                                    /*alpha=*/1.0f, /*beta=*/1.0f,
                                    /*trans_a=*/transA ? 1 : 0,
                                    /*trans_b=*/transB ? 1 : 0, output);
        } else {
            // Matmul without bias: Y = A @ B
            infini::ops::Gemm::Call(handle, config, a, b,
                                    /*alpha=*/1.0f, /*beta=*/0.0f,
                                    /*trans_a=*/transA ? 1 : 0,
                                    /*trans_b=*/transB ? 1 : 0, output);
        }
    }
};

REGISTER_ALL_DEVICES(OpType::MatMul, MatmulInfiniOpsKernel, "Matmul_InfiniOps");

} // namespace infini
