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

using DLTensorHolder = pair<DLTensor, Ref<vector<int64_t>>>;

class TVMRecordObj : public PerfRecordObj {
    // TODO: Add more attrs
  public:
    std::string kernelName;
    HashType simplifiedExprHash;
    std::string dllPath;
    std::string funcName;
    std::vector<int> inputIdx;
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
        tvm::runtime::PackedFunc packedFunc = getPackedFunction(tvmRecord->dllPath, tvmRecord->funcName);
        assert(packedFunc != nullptr);
        // ICHECK(packedFunc != nullptr);

        // prepare inputs and outputs
        vector<DLTensorHolder> inputsHolder;
        for (auto idx : tvmRecord->inputIdx) {
            inputsHolder.emplace_back(convertTensorToDLTensor(op->getInputs()[idx]));
        }
        DLTensorHolder outputHolder = convertTensorToDLTensor(op->getOutput());

        // make tvm arg and rv
        pair<vector<TVMValue>, vector<int>> preArgs = convertInOutToTVMArgs(inputsHolder, outputHolder);
        tvm::runtime::TVMRetValue rv;
        tvm::runtime::TVMArgs args(preArgs.first.data(), preArgs.second.data(), preArgs.first.size());

        packedFunc.CallPacked(args, &rv);

        // std::cout << "MemboundTVMPackedFunction::compute " << tvmRecord->dllPath << " "
        //             << tvmRecord->funcName << " " << tvmRecord->kernelName << std::endl;
        // std::cout << "Inputs: " << std::endl;
        // for (auto &in : op->getInputs()) {
        //     in->print();
        //     in->printData();
        // }
        // std::cout << "Output: " << std::endl;
        // op->getOutput()->print();
        // op->getOutput()->printData();
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

        const std::string func = "membound_" + std::to_string(hash);
        const std::string kernelName = func + "_kernel0";
	// string dllPath = "/home/hsh/InfiniTensor/build/.cache/generated_kernels/22494746/membound_22494746.so";
        auto dllPath = getAnsorDLL(
            inShapes, std::vector<std::string>(inShapes.size(), "float32"),
            outShape, "float32", stmts, func, op->toString(),
            expr->toReadable(), hash);
        
        // TODO: 1. Convert Tensor to DLTensor in convertTensorToDLTensor
        //       2. Store and load TVM function
        //       3. Prepare PerfRecordObj
        //       4. Impliment MemboundTVMPackedFunction::compute

        // remap input
        vector<int> inputIdx;
        int numInputs = op->getInputs().size();
        for (int i = 0; i < numInputs; ++i) {
            string inputName = visitor.getInputs()[i];
            int j = 0;
            for (; j < numInputs; ++j) {
                if (inputName == op->getNnetInputs()[j]->getName())
                    break;
            }
            inputIdx.emplace_back(j);
        }

        tvm::runtime::PackedFunc packedFunc = getPackedFunction(dllPath, func);
        assert(packedFunc != nullptr);
        // ICHECK(packedFunc != nullptr);

        // prepare inputs and outputs
        vector<DLTensorHolder> inputsHolder;
        for (auto idx : inputIdx) {
            inputsHolder.emplace_back(convertTensorToDLTensor(op->getInputs()[idx]));
        }
        DLTensorHolder outputHolder = convertTensorToDLTensor(op->getOutput());

        // make tvm arg and rv
        pair<vector<TVMValue>, vector<int>> preArgs = convertInOutToTVMArgs(inputsHolder, outputHolder);
        tvm::runtime::TVMRetValue rv;
        tvm::runtime::TVMArgs args(preArgs.first.data(), preArgs.second.data(), preArgs.first.size());

        // assert(inputsHolder.size() == 2);
        // op->getOutput()->print();
        // op->getInputs()[0]->print();
        // op->getInputs()[1]->print();
        // std::cout << inputsHolder[0].first.shape[0] << " " << inputsHolder[0].first.shape[1]
        //     << " " << inputsHolder[0].first.shape[2] << std::endl;
        // assert(inputsHolder[0].first.shape[1] == 3);
        // packedFunc(&outputHolder.first, &inputsHolder[0].first, &inputsHolder[1].first);

        ret->time = timeit(
            [&]() {
                packedFunc.CallPacked(args, &rv);
            },
            [&]() { context->sync(); });
        
        ret->kernelName = kernelName;
        ret->dllPath = dllPath;
        ret->funcName = func;
        ret->inputIdx = inputIdx;

        // std::cout << "MemboundTVMPackedFunction::tune " << ret->dllPath << " "
        //             << ret->funcName << " " << ret->kernelName << std::endl;
        // std::cout << "Inputs: " << std::endl;
        // for (auto &in : op->getInputs()) {
        //     in->print();
        //     in->printData();
        // }
        // std::cout << "Output: " << std::endl;
        // op->getOutput()->print();
        // op->getOutput()->printData();

        return std::dynamic_pointer_cast<PerfRecordObj>(ret);
    }

    /// @brief
    /// @param inDims
    /// @param inDTypes
    /// @param outDims
    /// @param outDType
    /// @param lambda
    /// @param funcName Generated function name
    /// @param nnetExpressionString Save expr in string for logging.
    /// @param nnetSimplifiedExprString Save simplified expr in string for
    /// logging.
    /// @param hashCode (optional) Hash code of the input expression for kernel
    /// cache.
    /// @return
    std::string getAnsorDLL(const std::vector<std::vector<int>> &inDims,
                 const std::vector<std::string> &inDTypes,
                 const std::vector<int> &outDims, const std::string &outDType,
                 const std::string &lambda, const std::string &funcName,
                 const std::string &nnetExprString,
                 const std::string &nnetSimplifiedExprString,
                 const HashType hashCode) const
    {
        std::string dllPath;
        try {
            start_interpreter();
            // Use static to avoid re-importing the module. Re-importing results
            // in cuBLAS failure, whose root cause is not identified yet.
            static auto func =
                py::module::import("cpp_plugin").attr("gen_ansor_so");
            py::tuple code =
                func(inDims, inDTypes, outDims, outDType, lambda, funcName,
                     nnetExprString, nnetSimplifiedExprString, std::to_string(hashCode));
            dllPath = py::str(code[0]);
        } catch (py::error_already_set &e) {
            if (e.matches(PyExc_ImportError)) {
                std::cerr << "Import Error. Don't forget to set environment "
                             "variable PYTHONPATH to contain "
                             "<repo-root>/python"
                          << std::endl;
            }
            throw;
        }

        return dllPath;
    }

    tvm::runtime::PackedFunc getPackedFunction(string path,
                                               string functionName) const {
        tvm::runtime::Module mod = tvm::runtime::Module::LoadFromFile(path);
        return mod.GetFunction(functionName);
    }

    DLTensorHolder
    convertTensorToDLTensor(const Tensor &tensor) const {
        IT_ASSERT_TODO(tensor->getRuntime()->isCuda());
        // The lifecycle of shapeInt64 is managed by the caller.
        auto shapeInt64 =
            make_ref<vector<int64_t>>();
        for (auto v : tensor->getDims())
            shapeInt64->push_back(v);
        // TODO
        DLTensor ret{
            .data = tensor->getRawDataPtr<void *>(),
            .device = DLDevice {.device_type = kDLCUDA, .device_id = 0},
            .ndim = (int32_t)shapeInt64->size(),
            .dtype = DLDataType {.code = (uint8_t)kDLFloat, .bits = 32, .lanes = 1},
            .shape = static_cast<int64_t *>(shapeInt64->data()),
            .strides = nullptr,
            .byte_offset = 0,
        };
        return {ret, shapeInt64};
    }

    pair<vector<TVMValue>, vector<int>>
    convertInOutToTVMArgs(const vector<DLTensorHolder> &inputs, const DLTensorHolder &output) const {
        vector<TVMValue> values;
        vector<int> type_codes;

        values.emplace_back(TVMValue {
            .v_handle = (void*)&output.first
        });
        type_codes.emplace_back(kTVMDLTensorHandle);

        for (auto &in : inputs) {
            values.emplace_back(TVMValue {
            .v_handle = (void*)&in.first
        });
        type_codes.emplace_back(kTVMDLTensorHandle);
        }

        return {values, type_codes};
    }
};

REGISTER_KERNEL(Device::CUDA, OpType::MemBound, DataType::Float32,
                MemboundTVMPackedFunction,
                "Memobund_TVM_Ansor_packed_funciton");
}; // namespace infini
