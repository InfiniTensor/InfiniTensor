#include "operators/matmul.h"
#include "core/kernel.h"
#include "cuda/cuda_runtime.h"
#include <chrono>
#include <functional>

namespace infini {
struct MatmulCudnnPerfRecordObj : public PerfRecordObj {
    cublasGemmAlgo_t algo = CUBLAS_GEMM_DEFAULT;
    json to_json () override {
        return json {{"time", this->time}, {"algo", this->algo}};
    }
    void from_json (json j) override {
        j.at("time").get_to(this->time);
        j.at("algo").get_to(this->algo);
    }
};
using MatmulCudnnPerfRecord = Ref<MatmulCudnnPerfRecordObj>;
constexpr int N_ALGO = 24;
constexpr cublasGemmAlgo_t ALGOS[N_ALGO] = {
    CUBLAS_GEMM_ALGO0,  CUBLAS_GEMM_ALGO1,  CUBLAS_GEMM_ALGO2,
    CUBLAS_GEMM_ALGO3,  CUBLAS_GEMM_ALGO4,  CUBLAS_GEMM_ALGO5,
    CUBLAS_GEMM_ALGO6,  CUBLAS_GEMM_ALGO7,  CUBLAS_GEMM_ALGO8,
    CUBLAS_GEMM_ALGO9,  CUBLAS_GEMM_ALGO10, CUBLAS_GEMM_ALGO11,
    CUBLAS_GEMM_ALGO12, CUBLAS_GEMM_ALGO13, CUBLAS_GEMM_ALGO14,
    CUBLAS_GEMM_ALGO15, CUBLAS_GEMM_ALGO16, CUBLAS_GEMM_ALGO17,
    CUBLAS_GEMM_ALGO18, CUBLAS_GEMM_ALGO19, CUBLAS_GEMM_ALGO20,
    CUBLAS_GEMM_ALGO21, CUBLAS_GEMM_ALGO22, CUBLAS_GEMM_ALGO23,
};

class matmulCublas : public Kernel {
    bool do_compute(const Operator &_op, const PerfRecord &_record,
                    const RuntimeObj *_context) const {
        auto op = as<MatmulObj>(_op);
        auto context = dynamic_cast<const CudaRuntimeObj *>(_context);
        void *const inAData = (op->getInputs(0)->getRawDataPtr<void *>());
        void *const inBData = (op->getInputs(1)->getRawDataPtr<void *>());
        void *const outData = (op->getOutput()->getRawDataPtr<void *>());
        auto record = as<MatmulCudnnPerfRecordObj>(_record);

        const auto [b, m, n, k] = op->getBMNK();
        auto opA =
            op->getTransA() ? CUBLAS_OP_T : CUBLAS_OP_N; // BLAS_N = col major
        auto opB = op->getTransB() ? CUBLAS_OP_T : CUBLAS_OP_N;
        const int lda = op->getTransA() ? m : k, ldb = op->getTransB() ? k : n,
                  ldc = n;
        const float alpha = 1.f, beta = 0.f;
        // TODO:use compute type
        cublasStatus_t stat;
        if (b > 1) {
            stat = cublasGemmStridedBatchedEx(
                context->cublasHandle(), opB, opA, n, m, k, &alpha, inBData,
                CUDA_R_32F, ldb, k * n, inAData, CUDA_R_32F, lda, m * k, &beta,
                outData, CUDA_R_32F, ldc, m * n, b, CUDA_R_32F, record->algo);
        } else {
            stat = cublasGemmEx(context->cublasHandle(), opB, opA, n, m, k,
                                &alpha, inBData, CUDA_R_32F, ldb, inAData,
                                CUDA_R_32F, lda, &beta, outData, CUDA_R_32F,
                                ldc, CUDA_R_32F, record->algo);
        }
        return (stat == CUBLAS_STATUS_SUCCESS);
    }

    void compute(const Operator &_op, const PerfRecord &_record,
                 const RuntimeObj *_context) const override {
        IT_ASSERT(do_compute(_op, _record, _context));
    }

    void compute(const Operator &op, const RuntimeObj *context) const override {
        auto record =
            make_ref<MatmulCudnnPerfRecordObj>(); // use default record;
        compute(op, record, context);
    }

    PerfRecord tune(const Operator &_op,
                    const RuntimeObj *_context) const override {
        auto context = dynamic_cast<const CudaRuntimeObj *>(_context);
        auto op = as<MatmulObj>(_op);
        auto ret = make_ref<MatmulCudnnPerfRecordObj>();
        ret->time = std::numeric_limits<double>::max();
        for (int i = 0; i < N_ALGO; i++) {
            auto rcd = make_ref<MatmulCudnnPerfRecordObj>();
            rcd->algo = ALGOS[i];
            if (!do_compute(_op, rcd, _context))
                continue;
            rcd->time = timeit([&]() { do_compute(_op, rcd, _context); },
                               [&]() { context->sync(); });
            if (rcd->time < ret->time)
                ret = rcd;
        }
        IT_ASSERT(ret->time < std::numeric_limits<double>::max(), "No valid "
                                                                  "algorithm "
                                                                  "found");
        return ret;
    }
};

REGISTER_KERNEL(Device::CUDA, OpType::Matmul, DataType::Float32, matmulCublas,
                "Matmul_cuBLAS_CUDA_Float32");

}; // namespace infini