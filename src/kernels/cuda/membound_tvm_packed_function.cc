#include "core/kernel.h"
#include "cuda/cuda_runtime.h"
#include "dlpack/dlpack.h"
#include "ffi/ffi_embed.h"
#include "nnet/Visitor/AsTVMVisitor.h"
#include "nnet/Visitor/HashVisitor.h"
#include "nnet/dbg.h"
#include "operators/membound.h"
#include "operators/pooling.h"
#include "tvm/runtime/module.h"
#include "tvm/runtime/packed_func.h"

namespace py = pybind11;

namespace infini {

class TVMRecordObj : public PerfRecordObj {
    // TODO: Add more attrs
  public:
    size_t logSize, ptxSize;
    std::string log, ptx;
    std::vector<int> invokeParams;
    std::string kernelName;
    HashType simplifiedExprHash;
};

using TVMRecord = Ref<TVMRecordObj>;

class MemboundTVMPackedFunction : public Kernel {
  public:
    void compute(const Operator &_op, const PerfRecord &record,
                 const RuntimeObj *_context) const override {
        auto op = as<MemBoundObj>(_op);
        // auto context = dynamic_cast<const CudaRuntimeObj *>(_context);
        auto tvmRecord = std::dynamic_pointer_cast<TVMRecordObj>(record);
        // TODO
    }

    void compute(const Operator &_op,
                 const RuntimeObj *_context) const override {
        IT_ASSERT(false, "A TVM record is required for membound kernel.");
    }

    std::string getVarName(const Tensor &t) const {
        return "var_" + std::to_string(t->getGuid());
    }

    // Premise: op is idempotent since it is called multiple times.
    PerfRecord tune(const Operator &_op,
                    const RuntimeObj *_context) const override {
        TVMRecord ret = std::make_shared<TVMRecordObj>();
        auto op = as<MemBoundObj>(_op);
        auto context = dynamic_cast<const CudaRuntimeObj *>(_context);

        // invoke Ansor to tune a membound kernel
        auto [expr, hash] = op->getSimplifiedNnetExpr();
        nnet::AsTVMVisitor visitor;
        visitor.dispatch(expr);
        auto &&stmts = visitor.getStmts();
        auto &&inShapes = visitor.getInputShapes();
        auto &&outShape = visitor.getOutputShape();

        std::vector<std::string> inputs;
        for (auto &&in : op->getInputs()) {
            inputs.emplace_back(getVarName(in));
        }
        const std::string output = getVarName(op->getOutput());

        const std::string func = "membound_" + std::to_string(hash);
        const std::string kernelName = func + "_kernel0";
        auto res = getAnsorCode(
            inShapes, std::vector<std::string>(inShapes.size(), "float32"),
            outShape, "float32", stmts, func, inputs, output, op->toString(),
            expr->toReadable(), hash);
        // TODO: 1. Convert Tensor to DLTensor in convertTensorToDLTensor
        //       2. Store and load TVM function
        //       3. Prepare PerfRecordObj
        //       4. Impliment MemboundTVMPackedFunction::compute
        return std::dynamic_pointer_cast<PerfRecordObj>(ret);
    }

    /// @brief
    /// @param inDims
    /// @param inDTypes
    /// @param outDims
    /// @param outDType
    /// @param lambda
    /// @param funcName Generated function name
    /// @param inputNames Input array names in the generated invocation code.
    /// @param outputName Output array names in the generated invocation code.
    /// @param nnetExpressionString Save expr in string for logging.
    /// @param nnetSimplifiedExprString Save simplified expr in string for
    /// logging.
    /// @param hashCode (optional) Hash code of the input expression for kernel
    /// cache.
    /// @return
    std::pair<std::string, std::vector<int>>
    getAnsorCode(const std::vector<std::vector<int>> &inDims,
                 const std::vector<std::string> &inDTypes,
                 const std::vector<int> &outDims, const std::string &outDType,
                 const std::string &lambda, const std::string &funcName,
                 const std::vector<std::string> &inputNames,
                 const std::string &outputName,
                 const std::string &nnetExprString,
                 const std::string &nnetSimplifiedExprString,
                 const HashType hashCode) const {
        std::string funcCode;
        std::vector<int> invokeParams;
        try {
            start_interpreter();
            // Use static to avoid re-importing the module. Re-importing results
            // in cuBLAS failure, whose root cause is not identified yet.
            static auto func =
                py::module::import("cpp_plugin").attr("gen_ansor_op");
            py::tuple code =
                func(inDims, inDTypes, outDims, outDType, lambda, funcName,
                     inputNames, outputName, nnetExprString,
                     nnetSimplifiedExprString, std::to_string(hashCode));
            funcCode = py::str(code[0]);
            auto temp = py::list(code[3]);
            for (int i = 0; i < 6; ++i) {
                invokeParams.push_back(temp[i].cast<int>());
            }
        } catch (py::error_already_set &e) {
            if (e.matches(PyExc_ImportError)) {
                std::cerr << "Import Error. Don't forget to set environment "
                             "variable PYTHONPATH to contain "
                             "<repo-root>/python"
                          << std::endl;
            }
            throw;
        }
        return std::make_pair(funcCode, invokeParams);
    }

    tvm::runtime::PackedFunc getPackedFunction(string path,
                                               string functionName) const {
        tvm::runtime::Module mod = tvm::runtime::Module::LoadFromFile(path);
        return mod.GetFunction(functionName);
    }

    pair<DLTensor, Ref<vector<int64_t>>>
    convertTensorToDLTensor(const Tensor &tensor) const {
        IT_ASSERT_TODO(tensor->getRuntime()->isCuda());
        // The lifecycle of shapeInt64 is managed by the caller.
        auto shapeInt64 =
            make_ref<vector<int64_t>>(tensor->getDims().size(), 0);
        for (auto v : tensor->getDims())
            shapeInt64->push_back(v);
        // TODO
        // DLTensor ret{
        //     .data = data->getPtr<void *>(),
        //     .device = kDLCUDA,
        //     .ndim = (int32_t)shape.size(),
        //     .dtype = kDLFloat,
        //     .shape = static_cast<int64_t *>(shapeInt64->data()),
        //     .strides = nullptr,
        //     .byte_offset = 0,
        // };
        // return {ret, shapeInt64};
    }
};

REGISTER_KERNEL(Device::CUDA, OpType::MemBound, DataType::Float32,
                MemboundTVMPackedFunction,
                "Memobund_TVM_Ansor_packed_funciton");
}; // namespace infini
