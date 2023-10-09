#include "operators/matmul.h"
#include "core/kernel.h"
#include "cuda/cuda_expand.h"
#include "cuda/cuda_runtime.h"
#include "utils/small_array.h"

namespace infini {

struct MatmulCublasPerfRecordObj : public PerfRecordObj {
    int algo = CUBLAS_GEMM_DEFAULT;
    void to_json(json &j) override {
        j["type"] = 2;
        j["data"] = std::make_pair(algo, time);
    }
    static PerfRecord from_json(const json &j) {
        MatmulCublasPerfRecordObj tmp;
        auto pr = j["data"].get<pair<int, double>>();
        tmp.algo = pr.first;
        tmp.time = pr.second;
        return make_ref<MatmulCublasPerfRecordObj>(tmp);
    }
};

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
        auto record = as<MatmulCublasPerfRecordObj>(_record);

        const auto [b, m, n, k] = op->getBMNK();
        auto opA =
            op->getTransA() ? CUBLAS_OP_T : CUBLAS_OP_N; // BLAS_N = col major
        auto opB = op->getTransB() ? CUBLAS_OP_T : CUBLAS_OP_N;
        const int lda = op->getTransA() ? m : k, ldb = op->getTransB() ? k : n,
                  ldc = n;
        float alpha = 1.f, beta = 0.f;
        if (op->numInputs() == 2) { // no bias
            beta = 0.f;
        } else { // broadcast bias to output
            beta = 1.f;
            auto inC = op->getInputs(2);
            auto out = op->getOutput();
            SmallArray inputShape, outputShape;
            int nDims = out->getRank();
            IT_ASSERT(nDims <= SMALL_ARRAY_SIZE);
            int outputsize = 1; // the length of the output vector after flatten
            int offset = nDims - inC->getRank();
            for (int i = 0; i < offset; ++i)
                inputShape.data[i] = 1;
            for (int i = 0; i < nDims; ++i) {
                outputShape.data[i] = out->getDims()[i];
                outputsize *= outputShape.data[i];
                if (i >= offset)
                    inputShape.data[i] = inC->getDims()[i - offset];
            }
            expandKernel(inC->getRawDataPtr<float *>(),
                         out->getRawDataPtr<float *>(), nDims, outputsize,
                         inputShape, outputShape);
        }
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
                context->cublasHandle(), opB, opA, n, m, k, &alpha, inBData,
                CUDA_R_32F, ldb, strideB, inAData, CUDA_R_32F, lda, strideA,
                &beta, outData, CUDA_R_32F, ldc, m * n, b, CUDA_R_32F,
                (cublasGemmAlgo_t)record->algo);
        } else {
            stat = cublasGemmEx(
                context->cublasHandle(), opB, opA, n, m, k, &alpha, inBData,
                CUDA_R_32F, ldb, inAData, CUDA_R_32F, lda, &beta, outData,
                CUDA_R_32F, ldc, CUDA_R_32F, (cublasGemmAlgo_t)record->algo);
        }
        // if (stat != CUBLAS_STATUS_SUCCESS)
        //     cout << cublasGetErrorString(stat);
        return (stat == CUBLAS_STATUS_SUCCESS);
    }

    void compute(const Operator &_op, const PerfRecord &_record,
                 const RuntimeObj *_context) const override {
        IT_ASSERT(do_compute(_op, _record, _context));
    }

    void compute(const Operator &op, const RuntimeObj *context) const override {
        auto record =
            make_ref<MatmulCublasPerfRecordObj>(); // use default record;
        compute(op, record, context);
    }

    PerfRecord tune(const Operator &_op,
                    const RuntimeObj *_context) const override {
        auto context = dynamic_cast<const CudaRuntimeObj *>(_context);
        auto op = as<MatmulObj>(_op);
        IT_ASSERT(context);
        IT_ASSERT(op);
        auto ret = make_ref<MatmulCublasPerfRecordObj>();
        ret->time = std::numeric_limits<double>::max();
        for (int i = 0; i < N_ALGO; i++) {
            auto rcd = make_ref<MatmulCublasPerfRecordObj>();
            rcd->algo = ALGOS[i];
            if (!do_compute(_op, rcd, _context))
                continue;
            rcd->time = timeit([&]() { do_compute(_op, rcd, _context); },
                               [&]() { context->sync(); });
            if (rcd->time < ret->time)
                ret = rcd;
        }
        IT_ASSERT(ret->time < std::numeric_limits<double>::max(),
                  "No valid algorithm found for " + op->toString());
        return ret;
    }
};

REGISTER_KERNEL(Device::CUDA, OpType::MatMul, DataType::Float32, matmulCublas,
                "Matmul_cuBLAS_CUDA_Float32");

REGISTER_CONSTRUCTOR(2, MatmulCublasPerfRecordObj::from_json);
}; // namespace infini
