#include "operators/logical.h"
#include "cuda/cuda_kernel_wihtout_config.h"
#include "cuda/cuda_logical.h"
#include "cuda/cuda_runtime.h"
#include "cuda/cuda_utility.h"

// Host-side wrappers that translate Operator/Tensor metadata into the
// corresponding CUDA kernel calls (declared in `cuda_logical.h`). These
// classes pack shape/dtype information and invoke the device kernels.

namespace infini {

class BinaryLogicalCuda : public CudaKernelWithoutConfig {
    void compute(const Operator &_op,
                 const RuntimeObj *_context) const override {
        auto op = as<BinaryLogicalObj>(_op);
        void *const aData = (op->getInputs(0)->getRawDataPtr<void *>());
        void *const bData = (op->getInputs(1)->getRawDataPtr<void *>());
        void *const cData = (op->getOutput()->getRawDataPtr<void *>());
        auto a_dim = op->getInputs(0)->getDims();
        auto b_dim = op->getInputs(1)->getDims();
        auto c_dim = op->getOutput()->getDims();
        const int dType = _op->getDType().getIndex();

        if (a_dim.size() > 4 || b_dim.size() > 4 || c_dim.size() > 4)
            IT_TODO_HALT();

        int a[4] = {1, 1, 1, 1};
        int b[4] = {1, 1, 1, 1};
        int c[4] = {1, 1, 1, 1};

        std::copy(a_dim.begin(), a_dim.end(), a + (4 - a_dim.size()));
        std::copy(b_dim.begin(), b_dim.end(), b + (4 - b_dim.size()));
        std::copy(c_dim.begin(), c_dim.end(), c + (4 - c_dim.size()));

        if (op->getOpType() == OpType::And) {
            And_kernel(dType, aData, bData, cData, a[0], a[1], a[2], a[3], b[0],
                       b[1], b[2], b[3], c[0], c[1], c[2], c[3]);
        } else if (op->getOpType() == OpType::Or) {
            Or_kernel(dType, aData, bData, cData, a[0], a[1], a[2], a[3], b[0],
                      b[1], b[2], b[3], c[0], c[1], c[2], c[3]);
        } else if (op->getOpType() == OpType::Xor) {
            Xor_kernel(dType, aData, bData, cData, a[0], a[1], a[2], a[3], b[0],
                       b[1], b[2], b[3], c[0], c[1], c[2], c[3]);
        } else if (op->getOpType() == OpType::BitwiseAnd) {
            BitAnd_kernel(dType, aData, bData, cData, a[0], a[1], a[2], a[3],
                          b[0], b[1], b[2], b[3], c[0], c[1], c[2], c[3]);
        } else if (op->getOpType() == OpType::BitwiseOr) {
            BitOr_kernel(dType, aData, bData, cData, a[0], a[1], a[2], a[3],
                         b[0], b[1], b[2], b[3], c[0], c[1], c[2], c[3]);
        } else if (op->getOpType() == OpType::BitwiseXor) {
            BitXor_kernel(dType, aData, bData, cData, a[0], a[1], a[2], a[3],
                          b[0], b[1], b[2], b[3], c[0], c[1], c[2], c[3]);
        } else if (op->getOpType() == OpType::BitLeftShift) {
            BitLeftShift_kernel(dType, aData, bData, cData, a[0], a[1], a[2],
                                a[3], b[0], b[1], b[2], b[3], c[0], c[1], c[2],
                                c[3]);
        } else if (op->getOpType() == OpType::BitRightShift) {
            BitRightShift_kernel(dType, aData, bData, cData, a[0], a[1], a[2],
                                 a[3], b[0], b[1], b[2], b[3], c[0], c[1], c[2],
                                 c[3]);
        }

        else {
            std::cerr << op->getOpType().toString() << " dtypeIndex=" << dType
                      << std::endl;
            IT_TODO_HALT();
        }
    }
};

class UnaryLogicalCuda : public CudaKernelWithoutConfig {
    void compute(const Operator &_op,
                 const RuntimeObj *_context) const override {
        auto op = as<UnaryLogicalObj>(_op);
        void *const aData = (op->getInputs(0)->getRawDataPtr<void *>());
        void *const bData = (op->getOutput()->getRawDataPtr<void *>());
        auto a_dim = op->getInputs(0)->getDims();
        auto b_dim = op->getOutput()->getDims();
        const int dType = _op->getDType().getIndex();

        if (a_dim.size() > 4)
            IT_TODO_HALT();

        int a[4] = {1, 1, 1, 1};
        int b[4] = {1, 1, 1, 1};

        std::copy(a_dim.begin(), a_dim.end(), a + (4 - a_dim.size()));
        std::copy(b_dim.begin(), b_dim.end(), b + (4 - b_dim.size()));

        if (op->getOpType() == OpType::Not) {
            Not_kernel(dType, aData, bData, a[0], a[1], a[2], a[3], b[0], b[1],
                       b[2], b[3]);
        } else if (op->getOpType() == OpType::BitwiseNot) {
            BitNot_kernel(dType, aData, bData, a[0], a[1], a[2], a[3], b[0],
                          b[1], b[2], b[3]);
        } else {
            std::cerr << op->getOpType().toString() << " dtypeIndex=" << dType
                      << std::endl;
            IT_TODO_HALT();
        }
    }
};

REGISTER_KERNEL(Device::CUDA, OpType::And, BinaryLogicalCuda, "And_CUDA");
REGISTER_KERNEL(Device::CUDA, OpType::Or, BinaryLogicalCuda, "Or_CUDA");
REGISTER_KERNEL(Device::CUDA, OpType::Xor, BinaryLogicalCuda, "Xor_CUDA");
REGISTER_KERNEL(Device::CUDA, OpType::BitwiseAnd, BinaryLogicalCuda,
                "And_CUDA");
REGISTER_KERNEL(Device::CUDA, OpType::BitwiseOr, BinaryLogicalCuda, "Or_CUDA");
REGISTER_KERNEL(Device::CUDA, OpType::BitwiseXor, BinaryLogicalCuda,
                "Xor_CUDA");
REGISTER_KERNEL(Device::CUDA, OpType::Not, UnaryLogicalCuda, "Not_CUDA");
REGISTER_KERNEL(Device::CUDA, OpType::BitwiseNot, UnaryLogicalCuda,
                "BitNot_CUDA");
REGISTER_KERNEL(Device::CUDA, OpType::BitLeftShift, BinaryLogicalCuda,
                "BitLeftShift_CUDA");
REGISTER_KERNEL(Device::CUDA, OpType::BitRightShift, BinaryLogicalCuda,
                "BitRightShift_CUDA");
}; // namespace infini
