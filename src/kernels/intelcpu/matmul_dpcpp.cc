#include "core/kernel.h"
#include "intelcpu/mkl_runtime.h"
#include "mkl.h"
#include "oneapi/mkl/blas.hpp"
#include "operators/matmul.h"
#include <CL/sycl.hpp>

namespace infini {
template <typename T> class MklDpcppMatmul : public CpuKernelWithoutConfig {
    void compute(const Operator &_op,
                 const RuntimeObj *context) const override {
        auto op = as<MatmulObj>(_op);
        // IT_ASSERT(op->getInputs().size() == 2, "Bias is not supported yet.");
        const T *A = op->getInputs(0)->getRawDataPtr<T *>();
        const T *B = op->getInputs(1)->getRawDataPtr<T *>();
        T *C = op->getOutput()->getRawDataPtr<T *>();
        IT_ASSERT(op->getAct() == ActType::None);
        const int m = op->getM(), n = op->getN(), k = op->getK(),
                  b = op->getB();

        auto opA = op->getTransA() ? oneapi::mkl::transpose::trans
                                   : oneapi::mkl::transpose::nontrans;
        auto opB = op->getTransB() ? oneapi::mkl::transpose::trans
                                   : oneapi::mkl::transpose::nontrans;
        // ldA is always a.col, and ldB is always b.col when row major
        const int ldA =
            std::max((opA == oneapi::mkl::transpose::nontrans) ? k : m, 1);
        const int ldB =
            std::max((opB == oneapi::mkl::transpose::nontrans) ? n : k, 1);
        const int ldC = std::max(n, 1);

        sycl::queue q(sycl::cpu_selector{});
        // Catch asynchronous exceptions
        auto exception_handler = [](cl::sycl::exception_list exceptions) {
            for (std::exception_ptr const &e : exceptions) {
                try {
                    std::rethrow_exception(e);
                } catch (cl::sycl::exception const &e) {
                    std::cout
                        << "Caught asynchronous SYCL exception during GEMM:\n"
                        << e.what() << std::endl;
                }
            }
        };

        // create execution queue and buffers of matrix data
        cl::sycl::queue main_queue(sycl::cpu_selector{}, exception_handler);

        cl::sycl::buffer<T, 1> A_buffer(A, op->getInputs(0)->size());
        cl::sycl::buffer<T, 1> B_buffer(B, op->getInputs(1)->size());

        cl::sycl::buffer<T, 1> O_buffer(C, op->getOutput(0)->size());

        // add oneapi::mkl::blas::gemm to execution queue
        try {
            if (op->getBeta() && op->getBias()) {
                // init C with bias
                IT_ASSERT_TODO(op->getBias()->size() ==
                               op->getOutput(0)->size());
                cl::sycl::buffer<T, 1> C_buffer(
                    op->getBias()->getRawDataPtr<T *>(), op->getBias()->size());
                oneapi::mkl::blas::row_major::copy(main_queue,
                                                   op->getBias()->size(),
                                                   C_buffer, 1, O_buffer, 1);
            }
            oneapi::mkl::blas::row_major::gemm_batch(
                main_queue, opA, opB, m, n, k, op->getAlpha(), A_buffer, ldA,
                m * k, B_buffer, ldB, k * n, op->getBeta(), O_buffer, ldC,
                m * n, b);
        } catch (cl::sycl::exception const &e) {
            std::cout << "\t\tCaught synchronous SYCL exception during GEMM:\n"
                      << e.what() << std::endl;
        }
    }
};

REGISTER_KERNEL(Device::INTELCPU, OpType::Matmul, DataType::Float32,
                MklDpcppMatmul<float>, "MklDpcppMatmul_CPU_float32");

} // namespace infini
