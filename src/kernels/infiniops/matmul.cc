#include "core/kernel.h"
#include "core/infiniops_bridge/tensor_convert.h"
#include "cpu/gemm/gemm.h"
#include "operators/matmul.h"

namespace infini {

class MatmulInfiniOpsKernel : public KernelWithoutConfig {
    void compute(const Operator &op, const RuntimeObj *context) const override {
        auto matmulOp = as<MatmulObj>(op);

        auto a = toInfiniOpsTensor(matmulOp->getInputs(0).get());
        auto b = toInfiniOpsTensor(matmulOp->getInputs(1).get());
        auto *out0 = matmulOp->getOutput().get();
        auto output = toInfiniOpsTensor(out0);

        infini::ops::Handle handle;
        infini::ops::Config config;

        bool transA = matmulOp->getTransA();
        bool transB = matmulOp->getTransB();

        if (auto bias = matmulOp->getBias()) {
            // Fused matmul + bias (element-wise): Y = alpha * A @ B + beta * C
            // 1. Copy bias data to output buffer
            // 2. Call Gemm with beta=1 to accumulate: output = A @ B + output
            auto *biasObj = bias.get();
            size_t bytes = out0->getBytes();
            std::memcpy(out0->getRawDataPtr<void *>(),
                        biasObj->getRawDataPtr<void *>(),
                        bytes);

            infini::ops::Gemm::Call(handle, config, a, b,
                                    /*alpha=*/1.0f, /*beta=*/1.0f,
                                    /*trans_a=*/transA ? 1 : 0,
                                    /*trans_b=*/transB ? 1 : 0,
                                    output);
        } else {
            // Matmul without bias: Y = A @ B
            infini::ops::Gemm::Call(handle, config, a, b,
                                    /*alpha=*/1.0f, /*beta=*/0.0f,
                                    /*trans_a=*/transA ? 1 : 0,
                                    /*trans_b=*/transB ? 1 : 0,
                                    output);
        }
    }
};

REGISTER_KERNEL(Device(Device::Type::kCpu), OpType::MatMul, MatmulInfiniOpsKernel,
                "Matmul_InfiniOps_CPU");

} // namespace infini
