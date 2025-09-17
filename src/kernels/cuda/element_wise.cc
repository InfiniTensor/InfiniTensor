#include "operators/element_wise.h"
#include "cuda/cuda_element_wise.h"
#include "cuda/cuda_kernel_wihtout_config.h"
#include "cuda/cuda_runtime.h"
#include "cuda/cuda_unary.h" //需要使用cast来转换数据类型
#include "cuda/cuda_utility.h"
void printTensorDesc(cudnnTensorDescriptor_t desc, const std::string &name) {
    cudnnDataType_t dataType;
    int nbDims;
    int dims[CUDNN_DIM_MAX];
    int strides[CUDNN_DIM_MAX];

    cudnnStatus_t status = cudnnGetTensorNdDescriptor(
        desc, CUDNN_DIM_MAX, &dataType, &nbDims, dims, strides);
    if (status != CUDNN_STATUS_SUCCESS) {
        std::cerr << "Failed to get tensor descriptor for " << name << ": "
                  << cudnnGetErrorString(status) << std::endl;
        return;
    }

    std::cout << "Tensor " << name << ":\n";
    std::cout << "  dtype: ";
    switch (dataType) {
    case CUDNN_DATA_FLOAT:
        std::cout << "Float32";
        break;
    case CUDNN_DATA_HALF:
        std::cout << "Float16";
        break;
    case CUDNN_DATA_INT32:
        std::cout << "Int32";
        break;
    case CUDNN_DATA_INT8:
        std::cout << "Int8";
        break;
    default:
        std::cout << "Other";
        break;
    }
    std::cout << "\n";

    std::cout << "  nbDims: " << nbDims << "\n";
    std::cout << "  dims: ";
    for (int i = 0; i < nbDims; i++)
        std::cout << dims[i] << " ";
    std::cout << "\n";

    std::cout << "  strides: ";
    for (int i = 0; i < nbDims; i++)
        std::cout << strides[i] << " ";
    std::cout << "\n";
}
namespace infini {
class ElementWiseCudnn : public CudaKernelWithoutConfig {
    virtual cudnnOpTensorOp_t getOpType() const = 0;
    virtual tuple<float, float, float> getAlphBeta() const {
        return {1.f, 1.f, 0.f};
    }
    void compute(const Operator &_op,
                 const RuntimeObj *_context) const override {
        auto op = as<ElementWiseObj>(_op);
        auto context = dynamic_cast<const CudaRuntimeObj *>(_context);

        auto aTensor = op->getInputs(0);
        auto bTensor = op->getInputs(1);
        auto cTensor = op->getOutput();

        // cudnnOpTensor only allows B to be broadcasted.
        if (aTensor->getDims() != cTensor->getDims()) {
            swap(aTensor, bTensor);
        }
        IT_ASSERT(aTensor->getDims() == cTensor->getDims(),
                  "Shape does not match.");

        void *const aData = (aTensor->getRawDataPtr<void *>());
        void *const bData = (bTensor->getRawDataPtr<void *>());
        void *const cData = (cTensor->getRawDataPtr<void *>());

        cudnnTensorDescriptor_t aDesc, bDesc, cDesc;
        // auto a_dim = aTensor->getDims();
        // auto b_dim = bTensor->getDims();
        // auto c_dim = cTensor->getDims();

        auto a_dim_base = aTensor->getDims();
        auto b_dim_base = bTensor->getDims();
        auto c_dim_base = cTensor->getDims();
        std::vector<int> a_dim;
        std::vector<int> b_dim;
        std::vector<int> c_dim;
        if (a_dim_base.size() > 4) {
            for (int d : a_dim_base) {
                if (d != 1) {
                    a_dim.push_back(d);
                }
            }
            if (a_dim.empty()) {
                a_dim.push_back(a_dim_base.back());
            }

        } else {
            a_dim = a_dim_base;
        }
        if (b_dim_base.size() > 4) {

            for (int d : b_dim_base) {
                if (d != 1) {
                    b_dim.push_back(d);
                }
            }
            if (b_dim.empty()) {
                b_dim.push_back(b_dim_base.back());
            }

        } else {
            b_dim = b_dim_base;
        }
        if (c_dim_base.size() > 4) {

            for (int d : c_dim_base) {
                if (d != 1) {
                    c_dim.push_back(d);
                }
            }
            if (c_dim.empty()) {
                c_dim.push_back(c_dim_base.back());
            }
        } else {
            c_dim = c_dim_base;
        }
        if (a_dim.size() > 4 || b_dim.size() > 4 || c_dim.size() > 4) {

            IT_TODO_HALT();
        }

        int a[4] = {1, 1, 1, 1};
        int b[4] = {1, 1, 1, 1};
        int c[4] = {1, 1, 1, 1};

        std::copy(a_dim.begin(), a_dim.end(), a + (4 - a_dim.size()));
        std::copy(b_dim.begin(), b_dim.end(), b + (4 - b_dim.size()));
        std::copy(c_dim.begin(), c_dim.end(), c + (4 - c_dim.size()));
        // bool condition = true;
        // for (size_t i = 0; i < 4; i++) {
        //     if (a[i] != b[i]) {
        //         condition = false;
        //         break;
        //     }
        // }
        // if (op->getOpType() != OpType::Div && op->getOpType() != OpType::Add
        // &&
        //     op->getOpType() != OpType::Pow && op->getOpType() !=
        //     OpType::Less) { condition = false;
        // }
        // if (condition) {
        //     int num = a[0] * a[1] * a[2] * a[3];
        //     const int dType = _op->getDType().getIndex();
        //     // std::cout << dType << std::endl;
        //     if (op->getOpType() == OpType::Div) {
        //         div_special_kernel(dType, aData, bData, cData, num);
        //     } else if (op->getOpType() == OpType::Add) {
        //         add_special_kernel(dType, aData, bData, cData, num);
        //     } else if (op->getOpType() == OpType::Pow) {
        //         pow_special_kernel(dType, aData, bData, cData, num);
        //     } else if (op->getOpType() == OpType::Less) {
        //         less_special_kernel(dType, aData, bData, cData, num);
        //     } else {
        //         IT_TODO_HALT();
        //     }
        // } else {
        //     auto cudnnDataType = cudnnDataTypeConvert(op->getDType());
        //     // get inputs
        //     checkCudnnError(cudnnCreateTensorDescriptor(&aDesc));
        //     checkCudnnError(cudnnSetTensor4dDescriptor(aDesc,
        //     CUDNN_TENSOR_NCHW,
        //                                                cudnnDataType, a[0],
        //                                                a[1], a[2], a[3]));

        //     checkCudnnError(cudnnCreateTensorDescriptor(&bDesc));
        //     checkCudnnError(cudnnSetTensor4dDescriptor(bDesc,
        //     CUDNN_TENSOR_NCHW,
        //                                                cudnnDataType, b[0],
        //                                                b[1], b[2], b[3]));

        //     // get outputs
        //     checkCudnnError(cudnnCreateTensorDescriptor(&cDesc));
        //     checkCudnnError(cudnnSetTensor4dDescriptor(cDesc,
        //     CUDNN_TENSOR_NCHW,
        //                                                cudnnDataType, c[0],
        //                                                c[1], c[2], c[3]));

        //     // get op descriptor
        //     cudnnOpTensorDescriptor_t opDesc;
        //     checkCudnnError(cudnnCreateOpTensorDescriptor(&opDesc));
        //     checkCudnnError(cudnnSetOpTensorDescriptor(
        //         opDesc, getOpType(), CUDNN_DATA_FLOAT,
        //         CUDNN_NOT_PROPAGATE_NAN));

        //     auto [aAlpha, bAlpha, beta] = getAlphBeta();

        //     checkCudnnError(cudnnOpTensor(context->cudnnHandle(), opDesc,
        //                                   &aAlpha, aDesc, aData, &bAlpha,
        //                                   bDesc, bData, &beta, cDesc,
        //                                   cData));

        //     // Destories in CUDA does not require sync. But cuDNN does not
        //     state
        //     // whether sync is required before destories.
        //     checkCudnnError(cudnnDestroyTensorDescriptor(aDesc));
        //     checkCudnnError(cudnnDestroyTensorDescriptor(bDesc));
        //     checkCudnnError(cudnnDestroyTensorDescriptor(cDesc));
        //     checkCudnnError(cudnnDestroyOpTensorDescriptor(opDesc));
        // }
        // auto cudnnDataType = cudnnDataTypeConvert(op->getDType());
        cudnnDataType_t cudnnDataType;
        int a_size = aTensor->size();
        int b_size = bTensor->size();
        int c_size = cTensor->size();
        float *aF = nullptr, *bF = nullptr, *cF = nullptr;
        cudaMalloc(&aF, a_size * sizeof(float));
        cudaMalloc(&bF, b_size * sizeof(float));
        cudaMalloc(&cF, c_size * sizeof(float));
        if (op->getDType() == DataType::Int32 ||
            op->getDType() == DataType::Int64 ||
            op->getDType() == DataType::UInt32 ||
            op->getDType() == DataType::UInt64) {
            cudnnDataType = CUDNN_DATA_FLOAT;

            if (op->getDType() == DataType::Int32) {
                cast_kernel<int32_t, float>((int32_t *)aData, (float *)aF,
                                            a_size);
                cast_kernel<int32_t, float>((int32_t *)bData, (float *)bF,
                                            b_size);
                cast_kernel<int32_t, float>((int32_t *)cData, (float *)cF,
                                            c_size);
            } else if (op->getDType() == DataType::Int64) {
                cast_kernel<int64_t, float>((int64_t *)aData, (float *)aF,
                                            a_size);
                cast_kernel<int64_t, float>((int64_t *)bData, (float *)bF,
                                            b_size);
                cast_kernel<int64_t, float>((int64_t *)cData, (float *)cF,
                                            c_size);
            } else if (op->getDType() == DataType::UInt32) {
                cast_kernel<uint32_t, float>((uint32_t *)aData, (float *)aF,
                                             a_size);
                cast_kernel<uint32_t, float>((uint32_t *)bData, (float *)bF,
                                             b_size);
                cast_kernel<uint32_t, float>((uint32_t *)cData, (float *)cF,
                                             c_size);
            } else if (op->getDType() == DataType::UInt64) {
                cast_kernel<uint64_t, float>((uint64_t *)aData, (float *)aF,
                                             a_size);
                cast_kernel<uint64_t, float>((uint64_t *)bData, (float *)bF,
                                             b_size);
                cast_kernel<uint64_t, float>((uint64_t *)cData, (float *)cF,
                                             c_size);
            }

        } else {
            cudnnDataType = cudnnDataTypeConvert(op->getDType());
        }
        // get inputs
        checkCudnnError(cudnnCreateTensorDescriptor(&aDesc));
        checkCudnnError(cudnnSetTensor4dDescriptor(
            aDesc, CUDNN_TENSOR_NCHW, cudnnDataType, a[0], a[1], a[2], a[3]));

        checkCudnnError(cudnnCreateTensorDescriptor(&bDesc));
        checkCudnnError(cudnnSetTensor4dDescriptor(
            bDesc, CUDNN_TENSOR_NCHW, cudnnDataType, b[0], b[1], b[2], b[3]));

        // get outputs
        checkCudnnError(cudnnCreateTensorDescriptor(&cDesc));
        checkCudnnError(cudnnSetTensor4dDescriptor(
            cDesc, CUDNN_TENSOR_NCHW, cudnnDataType, c[0], c[1], c[2], c[3]));

        // get op descriptor
        cudnnOpTensorDescriptor_t opDesc;
        checkCudnnError(cudnnCreateOpTensorDescriptor(&opDesc));
        checkCudnnError(cudnnSetOpTensorDescriptor(
            opDesc, getOpType(), CUDNN_DATA_FLOAT, CUDNN_NOT_PROPAGATE_NAN));
        auto [aAlpha, bAlpha, beta] = getAlphBeta();
        if (op->getDType() == DataType::Int32 ||
            op->getDType() == DataType::Int64 ||
            op->getDType() == DataType::UInt32 ||
            op->getDType() == DataType::UInt64) {
            cudnnDataType = CUDNN_DATA_FLOAT;

            if (op->getDType() == DataType::Int32) {
                checkCudnnError(cudnnOpTensor(context->cudnnHandle(), opDesc,
                                              &aAlpha, aDesc, aF, &bAlpha,
                                              bDesc, bF, &beta, cDesc, cF));
                cast_kernel<float, int32_t>((float *)cF, (int32_t *)cData,
                                            c_size);
            } else if (op->getDType() == DataType::Int64) {
                checkCudnnError(cudnnOpTensor(context->cudnnHandle(), opDesc,
                                              &aAlpha, aDesc, aF, &bAlpha,
                                              bDesc, bF, &beta, cDesc, cF));
                cast_kernel<float, int64_t>((float *)cF, (int64_t *)cData,
                                            c_size);
            } else if (op->getDType() == DataType::UInt32) {
                checkCudnnError(cudnnOpTensor(context->cudnnHandle(), opDesc,
                                              &aAlpha, aDesc, aF, &bAlpha,
                                              bDesc, bF, &beta, cDesc, cF));
                cast_kernel<float, uint32_t>((float *)cF, (uint32_t *)cData,
                                             c_size);
            } else if (op->getDType() == DataType::UInt64) {
                checkCudnnError(cudnnOpTensor(context->cudnnHandle(), opDesc,
                                              &aAlpha, aDesc, aF, &bAlpha,
                                              bDesc, bF, &beta, cDesc, cF));
                cast_kernel<float, uint64_t>((float *)cF, (uint64_t *)cData,
                                             c_size);
            }

        } else {
            checkCudnnError(cudnnOpTensor(context->cudnnHandle(), opDesc,
                                          &aAlpha, aDesc, aData, &bAlpha, bDesc,
                                          bData, &beta, cDesc, cData));
        }
        cudaFree(aF);
        cudaFree(bF);
        cudaFree(cF);
        // Destories in CUDA does not require sync. But cuDNN does not state
        // whether sync is required before destories.
        checkCudnnError(cudnnDestroyTensorDescriptor(aDesc));
        checkCudnnError(cudnnDestroyTensorDescriptor(bDesc));
        checkCudnnError(cudnnDestroyTensorDescriptor(cDesc));
        checkCudnnError(cudnnDestroyOpTensorDescriptor(opDesc));
    }

    void compute(const Operator &op, const PerfRecord &record,
                 const RuntimeObj *context) const override {
        compute(op, context);
    }

  public:
    ElementWiseCudnn() {
        ComputeFuncPtr computePtr = [this](const Operator &op,
                                           const PerfRecord &record,
                                           const RuntimeObj *context) {
            this->compute(op, record, context);
        };
        funcVec.emplace_back(computePtr);
    }
};

class AddCudnn : public ElementWiseCudnn {
    cudnnOpTensorOp_t getOpType() const override { return CUDNN_OP_TENSOR_ADD; }
};

class SubCudnn : public ElementWiseCudnn {
    cudnnOpTensorOp_t getOpType() const override { return CUDNN_OP_TENSOR_ADD; }
    tuple<float, float, float> getAlphBeta() const override {
        return {1.f, -1.f, 0.f};
    }
};

class MulCudnn : public ElementWiseCudnn {
    cudnnOpTensorOp_t getOpType() const override { return CUDNN_OP_TENSOR_MUL; }
};

class MinCudnn : public ElementWiseCudnn {
    cudnnOpTensorOp_t getOpType() const override { return CUDNN_OP_TENSOR_MIN; }
};

class MaxCudnn : public ElementWiseCudnn {
    cudnnOpTensorOp_t getOpType() const override { return CUDNN_OP_TENSOR_MAX; }
};

class ElementWiseCuda : public CudaKernelWithoutConfig {
    void compute(const Operator &_op,
                 const RuntimeObj *_context) const override {
        auto op = as<ElementWiseObj>(_op);
        void *const aData = (op->getInputs(0)->getRawDataPtr<void *>());
        void *const bData = (op->getInputs(1)->getRawDataPtr<void *>());
        void *const cData = (op->getOutput()->getRawDataPtr<void *>());
        auto a_dim = op->getInputs(0)->getDims();
        auto b_dim = op->getInputs(1)->getDims();
        auto c_dim = op->getOutput()->getDims();
        const int dType = _op->getDType().getIndex();

        // Use optimized kernel if b is constant
        if (b_dim.size() == 0) {
            if (op->getOpType() == OpType::Div) {
                div_const_kernel(dType, aData, bData, cData,
                                 op->getOutput()->size());
                return;
            } else if (op->getOpType() == OpType::Pow) {
                pow_const_kernel(dType, aData, bData, cData,
                                 op->getOutput()->size());
                return;
            }
        }

        if (a_dim.size() > 4 || b_dim.size() > 4 || c_dim.size() > 4)
            IT_TODO_HALT();

        int a[4] = {1, 1, 1, 1};
        int b[4] = {1, 1, 1, 1};
        int c[4] = {1, 1, 1, 1};

        std::copy(a_dim.begin(), a_dim.end(), a + (4 - a_dim.size()));
        std::copy(b_dim.begin(), b_dim.end(), b + (4 - b_dim.size()));
        std::copy(c_dim.begin(), c_dim.end(), c + (4 - c_dim.size()));

        if (op->getOpType() == OpType::Div) {
            div_kernel(dType, aData, bData, cData, a[0], a[1], a[2], a[3], b[0],
                       b[1], b[2], b[3], c[0], c[1], c[2], c[3]);
        } else if (op->getOpType() == OpType::Add) {
            add_kernel(dType, aData, bData, cData, a[0], a[1], a[2], a[3], b[0],
                       b[1], b[2], b[3], c[0], c[1], c[2], c[3]);
        } else if (op->getOpType() == OpType::Pow) {
            pow_kernel(dType, aData, bData, cData, a[0], a[1], a[2], a[3], b[0],
                       b[1], b[2], b[3], c[0], c[1], c[2], c[3]);
        } else if (op->getOpType() == OpType::Less) {
            less_kernel(dType, aData, bData, cData, a[0], a[1], a[2], a[3],
                        b[0], b[1], b[2], b[3], c[0], c[1], c[2], c[3]);
        } else if (op->getOpType() == OpType::Equal) {
            equal_kernel(dType, aData, bData, cData, a[0], a[1], a[2], a[3],
                         b[0], b[1], b[2], b[3], c[0], c[1], c[2], c[3]);
        } else {
            IT_TODO_HALT();
        }
    }
};

class ElementWiseLogicCuda : public CudaKernelWithoutConfig {
    void compute(const Operator &_op,
                 const RuntimeObj *_context) const override {
        auto op = as<ElementWiseLogicObj>(_op);
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

        if (op->getOpType() == OpType::Less) {
            less_kernel(dType, aData, bData, cData, a[0], a[1], a[2], a[3],
                        b[0], b[1], b[2], b[3], c[0], c[1], c[2], c[3]);
        } else if (op->getOpType() == OpType::Equal) {
            equal_kernel(dType, aData, bData, cData, a[0], a[1], a[2], a[3],
                         b[0], b[1], b[2], b[3], c[0], c[1], c[2], c[3]);
        } else {
            IT_TODO_HALT();
        }
    }
};

REGISTER_KERNEL(Device::CUDA, OpType::Add, AddCudnn, "Add_cuDNN_CUDA");
REGISTER_KERNEL(Device::CUDA, OpType::Sub, SubCudnn, "Sub_cuDNN_CUDA");
REGISTER_KERNEL(Device::CUDA, OpType::Mul, MulCudnn, "Mul_cuDNN_CUDA");
REGISTER_KERNEL(Device::CUDA, OpType::Min, MinCudnn, "Min_cuDNN_CUDA");
REGISTER_KERNEL(Device::CUDA, OpType::Max, MaxCudnn, "Max_cuDNN_CUDA");

REGISTER_KERNEL(Device::CUDA, OpType::Div, ElementWiseCuda, "Div_CUDA");
REGISTER_KERNEL(Device::CUDA, OpType::Pow, ElementWiseCuda, "Pow_CUDA");
REGISTER_KERNEL(Device::CUDA, OpType::Less, ElementWiseLogicCuda, "Less_CUDA");
REGISTER_KERNEL(Device::CUDA, OpType::Equal, ElementWiseLogicCuda,
                "Equal_CUDA");

}; // namespace infini
