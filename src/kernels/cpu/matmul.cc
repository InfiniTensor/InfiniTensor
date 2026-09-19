#include "operators/matmul.h"
#include "core/kernel.h"
#include "utils/operator_utils.h"
#include <numeric>

namespace infini {

class NaiveMatmul : public CpuKernelWithoutConfig {
    /// How many matrices an operand carries, being the product of everything
    /// before its last two dimensions.
    static int batchOf(const Tensor &tensor) {
        const auto &dims = tensor->getDims();
        if (dims.size() <= 2) {
            return 1;
        }
        return std::accumulate(dims.begin(), dims.end() - 2, 1,
                               std::multiplies<int>());
    }

    template <typename T>
    void doCompute(const Operator &_op, const RuntimeObj *context) const {
        auto op = as<MatmulObj>(_op);
        T *A = op->getInputs(0)->getRawDataPtr<T *>();
        T *B = op->getInputs(1)->getRawDataPtr<T *>();
        T *C = op->getOutput()->getRawDataPtr<T *>();
        IT_ASSERT(op->getAct() == ActType::None);
        const int M = op->getM(), N = op->getN(), K = op->getK();
        // A transpose changes only the last two logical indices. The stored
        // matrix size (and hence its batch stride) stays M*K or K*N.
        const int rowStrideA = op->getTransA() ? 1 : K;
        const int reductionStrideA = op->getTransA() ? M : 1;
        const int reductionStrideB = op->getTransB() ? 1 : N;
        const int columnStrideB = op->getTransB() ? K : 1;

        // Everything before the last two dimensions is a stack of matrices to
        // work through. The operator has already broadcast those dimensions
        // together and reports the total, so the loop follows its count rather
        // than working the shapes out again.
        const int batch = op->getB();
        const int batchA = batchOf(op->getInputs(0));
        const int batchB = batchOf(op->getInputs(1));
        // An operand either carries a matrix for every one of them or a single
        // matrix meant for all of them. Anything else would be a broadcast this
        // does not describe, and reading it as either would be wrong.
        IT_ASSERT(batchA == batch || batchA == 1);
        IT_ASSERT(batchB == batch || batchB == 1);
        // A single matrix is reread for each one, which is what a zero stride
        // says.
        const int strideA = batchA == 1 ? 0 : M * K;
        const int strideB = batchB == 1 ? 0 : K * N;

        // A `Gemm` carries the bias of the layer it came from as a third
        // operand, and a simplifier will fuse a matmul and its following add
        // into one, so refusing a bias here refuses an ordinary linear layer.
        // It is added over the last two dimensions of the result, broadcast the
        // way any operand is, which is usually one value per output column.
        const bool hasBias = op->getInputs().size() > 2;
        const T *bias =
            hasBias ? op->getInputs(2)->getRawDataPtr<T *>() : nullptr;
        // The same right-aligned broadcast the element-wise kernel does, over
        // the matrix the bias is added to rather than over the whole stack: a
        // bias is a property of the layer, so every matrix in the batch gets
        // the same one.
        const Shape matrix{M, N};
        Shape biasShape(matrix.size(), 1);
        Shape biasStride(matrix.size(), 0);
        if (hasBias) {
            const auto &dims = op->getInputs(2)->getDims();
            IT_ASSERT(dims.size() <= matrix.size(),
                      "a bias may not have more dimensions than the matrix it "
                      "is added to");
            std::copy(dims.begin(), dims.end(),
                      biasShape.begin() + (matrix.size() - dims.size()));
            for (size_t i = 0; i < matrix.size(); ++i) {
                IT_ASSERT(biasShape[i] == matrix[i] || biasShape[i] == 1,
                          "a bias dimension must match the matrix or be one");
            }
            int p = 1;
            for (auto i = matrix.size(); i > 0; --i) {
                biasStride[i - 1] = p;
                p = p * biasShape[i - 1];
            }
        }

        for (int p = 0; p < batch; p++) {
            const T *a = A + static_cast<ptrdiff_t>(p) * strideA;
            const T *b = B + static_cast<ptrdiff_t>(p) * strideB;
            T *c = C + static_cast<ptrdiff_t>(p) * M * N;
            for (int i = 0; i < M; i++) {
                for (int j = 0; j < N; j++) {
                    T sum = 0;
                    for (int k = 0; k < K; k++) {
                        sum += a[i * rowStrideA + k * reductionStrideA] *
                               b[k * reductionStrideB + j * columnStrideB];
                    }
                    if (hasBias) {
                        sum += bias[delocate_index({i, j}, biasShape,
                                                   biasStride)];
                    }
                    c[i * N + j] = sum;
                }
            }
        }
    }

    void compute(const Operator &_op,
                 const RuntimeObj *context) const override {
#define CASE(N)                                                                \
    case N:                                                                    \
        doCompute<DT<N>::t>(_op, context)

        int dataTypeIdx = _op->getDType().getIndex();
        switch (dataTypeIdx) {
            CASE(1); // DataType::Float32
            break;
            CASE(12); // DataType::UInt32
            break;
        default:
            IT_TODO_HALT();
        }
    }
};

REGISTER_KERNEL(Device::CPU, OpType::MatMul, NaiveMatmul, "MatmulNaive_CPU");

} // namespace infini
