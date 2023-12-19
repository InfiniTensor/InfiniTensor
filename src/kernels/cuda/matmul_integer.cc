#include "operators/matmul_integer.h"
#include "cuda/cuda_kernel_wihtout_config.h"
#include "cuda/cuda_matmul_integer.h"
#include "cuda/cuda_runtime.h"
#include "utils/small_array.h"
#include <thrust/transform.h>

namespace infini {

class matmulIntegerCublas : public CudaKernelWithoutConfig {
    bool do_compute(const Operator &_op, const RuntimeObj *_context) const {
        auto op = as<MatmulIntegerObj>(_op);
        auto context = dynamic_cast<const CudaRuntimeObj *>(_context);
        void *const inAData = (op->getInputs(0)->getRawDataPtr<void *>());
        void *const inBData = (op->getInputs(1)->getRawDataPtr<void *>());
        void *const outData = (op->getOutput()->getRawDataPtr<void *>());

        const auto [b, m, n, k] = op->getBMNK();
        if (op->numInputs() >= 3) { // have a_zero_point
            int aZeroSize = op->getInputs(2)->size();
            int aSize = op->getInputs(0)->size();
            void *const aZeroPointData =
                (op->getInputs(2)->getRawDataPtr<void *>());
            if (op->getInputs(0)->getDType() == DataType::Int8) {
                if (aZeroSize > 1) {
                    subA_kernel(DataType::Int8.getIndex(), inAData,
                                aZeroPointData, aSize, k, 1);
                } else {
                    subA_kernel(DataType::Int8.getIndex(), inAData,
                                aZeroPointData, aSize, k, 0);
                }
            }
            if (op->getInputs(0)->getDType() == DataType::UInt8) {
                if (aZeroSize > 1) {
                    subA_kernel(DataType::UInt8.getIndex(), inAData,
                                aZeroPointData, aSize, k, 1);
                } else {
                    subA_kernel(DataType::UInt8.getIndex(), inAData,
                                aZeroPointData, aSize, k, 0);
                }
            }
        }
        if (op->numInputs() == 4) { // have b_zero_point
            int bZeroSize = op->getInputs(3)->size();
            int bSize = op->getInputs(1)->size();
            void *const bZeroPointData =
                (op->getInputs(3)->getRawDataPtr<void *>());
            if (op->getInputs(1)->getDType() == DataType::Int8) {
                if (bZeroSize > 1) {
                    subB_kernel(DataType::Int8.getIndex(), inBData,
                                bZeroPointData, bSize, k, n, 1);
                } else {
                    subB_kernel(DataType::Int8.getIndex(), inBData,
                                bZeroPointData, bSize, k, n, 0);
                }
            }
            if (op->getInputs(1)->getDType() == DataType::UInt8) {
                if (bZeroSize > 1) {
                    subB_kernel(DataType::UInt8.getIndex(), inBData,
                                bZeroPointData, bSize, k, n, 1);
                } else {
                    subB_kernel(DataType::UInt8.getIndex(), inBData,
                                bZeroPointData, bSize, k, n, 0);
                }
            }
        }
        int lda = k, ldb = n, ldc = n;
        int32_t alpha = 1, beta = 0;

        // TODO:use compute type
        cublasStatus_t stat;
        if (b > 1) {
            // Support batch broadcast with zero stride
            int dimA = op->getInputs(0)->getRank();
            int dimB = op->getInputs(1)->getRank();
            long long strideA =
                (dimA == 2 ||
                 (dimA == 3 && op->getInputs(0)->getDims()[0] == 1))
                    ? 0 // Broadcast the batch dimension if batch size is 1
                    : m * k;
            long long strideB =
                (dimB == 2 ||
                 (dimB == 3 && op->getInputs(1)->getDims()[0] == 1))
                    ? 0 // Broadcast the batch dimension if batch size is 1
                    : n * k;
            stat = cublasGemmStridedBatchedEx(
                context->cublasHandle(), CUBLAS_OP_N, CUBLAS_OP_N, n, m, k,
                &alpha, inBData, CUDA_R_8I, ldb, strideB, inAData, CUDA_R_8I,
                lda, strideA, &beta, outData, CUDA_R_32I, ldc, m * n, b,
                CUDA_R_32I, CUBLAS_GEMM_DEFAULT);
        } else {
            stat = cublasGemmEx(
                context->cublasHandle(), CUBLAS_OP_N, CUBLAS_OP_N, n, m, k,
                &alpha, inBData, CUDA_R_8I, ldb, inAData, CUDA_R_8I, lda, &beta,
                outData, CUDA_R_32I, ldc, CUDA_R_32I, CUBLAS_GEMM_DEFAULT);
        }
        return (stat == CUBLAS_STATUS_SUCCESS);
    }

    void compute(const Operator &_op,
                 const RuntimeObj *_context) const override {
        IT_ASSERT(do_compute(_op, _context));
    }
};

REGISTER_KERNEL(Device::CUDA, OpType::MatMulInteger, matmulIntegerCublas,
                "MatmulInteger_cuBLAS_CUDA");

}; // namespace infini
