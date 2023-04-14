#ifdef INFINI_USE_TVM
#include "core/kernel.h"
#include "cuda/cuda_runtime.h"
#include "dlpack/dlpack.h"
#include "ffi/ffi_embed.h"
#include "nnet/Visitor/AsTVMVisitor.h"
#include "operators/membound.h"
#include "operators/pooling.h"
#include "tvm/runtime/module.h"
#include "tvm/runtime/packed_func.h"
#include <nlohmann/json.hpp>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>
using json = nlohmann::json;

namespace py = pybind11;

namespace infini {

using DLTensorHolder = pair<DLTensor, Ref<vector<int64_t>>>;

class TVMRecordObj : public PerfRecordObj {
  public:
    std::string kernelName;
    HashType simplifiedExprHash;
    std::string dllPath;
    std::string funcName;
    std::vector<int> inputIdx;
    tvm::runtime::PackedFunc packedFunc;
};

using TVMRecord = Ref<TVMRecordObj>;

class MemboundTVMPackedFunction : public Kernel {
  public:
    void compute(const Operator &_op, const PerfRecord &record,
                 const RuntimeObj *_context) const override {
        auto op = as<MemBoundObj>(_op);
        // auto context = dynamic_cast<const CudaRuntimeObj *>(_context);
        auto tvmRecord = std::dynamic_pointer_cast<TVMRecordObj>(record);
        tvm::runtime::PackedFunc packedFunc = tvmRecord->packedFunc;
        // IT_ASSERT(packedFunc != nullptr);

        // prepare inputs and outputs
        vector<DLTensorHolder> inputsHolder;
        for (auto idx : tvmRecord->inputIdx) {
            inputsHolder.emplace_back(
                convertTensorToDLTensor(op->getInputs()[idx]));
        }
        DLTensorHolder outputHolder = convertTensorToDLTensor(op->getOutput());

        // make tvm arg and rv
        pair<vector<TVMValue>, vector<int>> preArgs =
            convertInOutToTVMArgs(inputsHolder, outputHolder);
        tvm::runtime::TVMRetValue rv;
        tvm::runtime::TVMArgs args(preArgs.first.data(), preArgs.second.data(),
                                   preArgs.first.size());

        packedFunc.CallPacked(args, &rv);
    }

    void compute(const Operator &_op,
                 const RuntimeObj *_context) const override {
        IT_ASSERT(false, "A TVM record is required for membound kernel.");
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
        // Set the dllPath directly when debugging
        auto dllPath = getAnsorDLL(
            inShapes, std::vector<std::string>(inShapes.size(), "float32"),
            outShape, "float32", stmts, func, op->toString(),
            expr->toReadable(), hash);

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
            IT_ASSERT(j < numInputs, "Cannot find input name: " + inputName);
            inputIdx.emplace_back(j);
        }

        tvm::runtime::PackedFunc packedFunc = getPackedFunction(dllPath, func);
        IT_ASSERT(packedFunc != nullptr);

        // prepare inputs and outputs
        vector<DLTensorHolder> inputsHolder;
        for (auto idx : inputIdx) {
            inputsHolder.emplace_back(
                convertTensorToDLTensor(op->getInputs()[idx]));
        }
        DLTensorHolder outputHolder = convertTensorToDLTensor(op->getOutput());

        // make tvm arg and rv
        pair<vector<TVMValue>, vector<int>> preArgs =
            convertInOutToTVMArgs(inputsHolder, outputHolder);
        tvm::runtime::TVMRetValue rv;
        tvm::runtime::TVMArgs args(preArgs.first.data(), preArgs.second.data(),
                                   preArgs.first.size());

        ret->time = timeit([&]() { packedFunc.CallPacked(args, &rv); },
                           [&]() { context->sync(); });
        ret->kernelName = kernelName;
        ret->dllPath = dllPath;
        ret->funcName = func;
        ret->inputIdx = inputIdx;
        ret->packedFunc = packedFunc;

        return std::dynamic_pointer_cast<PerfRecordObj>(ret);
    }

    /// @brief Invoke TVM with pybind11. This approach is deprecated since it
    /// ruins the python global scope and somehow breaks the cuBLAS runtime.
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
    std::string getAnsorDLLPybind11(const std::vector<std::vector<int>> &inDims,
                                    const std::vector<std::string> &inDTypes,
                                    const std::vector<int> &outDims,
                                    const std::string &outDType,
                                    const std::string &lambda,
                                    const std::string &funcName,
                                    const std::string &nnetExprString,
                                    const std::string &nnetSimplifiedExprString,
                                    const HashType hashCode) const {
        std::string dllPath;
        try {
            start_interpreter();
            // Use static to avoid re-importing the module. Re-importing results
            // in cuBLAS failure, whose root cause is not identified yet.
            static auto func =
                py::module::import("cpp_plugin").attr("gen_ansor_so");
            py::tuple code =
                func(inDims, inDTypes, outDims, outDType, lambda, funcName,
                     nnetExprString, nnetSimplifiedExprString,
                     std::to_string(hashCode));
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

    std::string serializeTVMArgs(const std::vector<std::vector<int>> &inDims,
                                 const std::vector<std::string> &inDTypes,
                                 const std::vector<int> &outDims,
                                 const std::string &outDType,
                                 const std::string &lambda,
                                 const std::string &funcName,
                                 const std::string &nnetExprString,
                                 const std::string &nnetSimplifiedExprString,
                                 const HashType hashCode) const {
        json j;
        // Consistant with python API interface
        j["input_tensors"] = inDims;
        j["input_dtypes"] = inDTypes;
        j["output_tensor"] = outDims;
        j["output_dtype"] = outDType;
        j["tvm_code"] = lambda;
        j["func_name"] = funcName;
        j["nnet_expression"] = nnetExprString;
        j["nnet_simplified_expression"] = nnetSimplifiedExprString;
        j["hash_code"] = std::to_string(hashCode);
        return j.dump();
    }

    std::string getAnsorDLL(const std::vector<std::vector<int>> &inDims,
                            const std::vector<std::string> &inDTypes,
                            const std::vector<int> &outDims,
                            const std::string &outDType,
                            const std::string &lambda,
                            const std::string &funcName,
                            const std::string &nnetExprString,
                            const std::string &nnetSimplifiedExprString,
                            const HashType hashCode) const {
        int fdP2C[2], fdC2P[2];
        for (auto fd : {fdP2C, fdC2P}) {
            int status = pipe(fd);
            IT_ASSERT(status == 0, "pipe failed");
        }
        pid_t pid = fork();
        IT_ASSERT(pid >= 0, "fork failed");
        if (pid == 0) { // Child process
            close(fdP2C[1]);
            close(fdC2P[0]);

            dup2(fdP2C[0], STDIN_FILENO);
            close(fdP2C[0]);

            string cmd =
                "from cpp_plugin.gen_ansor_so import pipe_gen; pipe_gen(+" +
                std::to_string(fdC2P[1]) + ")";
            const char *const argv[] = {"python3", "-c", cmd.data(), NULL};
            execvp("python3", const_cast<char *const *>(argv));
        } else { // Parent process
            close(fdP2C[0]);
            close(fdC2P[1]);

            // Write to pipe
            string serializedArgs = serializeTVMArgs(
                inDims, inDTypes, outDims, outDType, lambda, funcName,
                nnetExprString, nnetSimplifiedExprString, hashCode);
            int status = -1;
            status =
                write(fdP2C[1], serializedArgs.data(), serializedArgs.size());
            IT_ASSERT((size_t)status == serializedArgs.size(),
                      "Failed to write to pipe");
            close(fdP2C[1]);

            // Wait for TVM
            waitpid(pid, &status, 0);
            IT_ASSERT(WIFEXITED(status), "TVM process was terminated");
            const int es = WEXITSTATUS(status);
            IT_ASSERT(es == 0,
                      "TVM process exit with code " + std::to_string(es));

            // Read from pipe
            FILE *stream;
            stream = fdopen(fdC2P[0], "r");
            char buf_read[257] = {0};
            status = std::fscanf(stream, "%256c", buf_read);
            IT_ASSERT(status == 1, "Failed to read from pipe");
            IT_ASSERT(buf_read[256] == 0, "Pipe buffer overflow");
            fclose(stream);
            close(fdC2P[0]);
            return buf_read;
        }
        IT_ASSERT(false, "Should not reach here");
        return "";
    }

    tvm::runtime::PackedFunc getPackedFunction(string path,
                                               string functionName) const {
        tvm::runtime::Module mod = tvm::runtime::Module::LoadFromFile(path);
        return mod.GetFunction(functionName);
    }

    DLTensorHolder convertTensorToDLTensor(const Tensor &tensor) const {
        IT_ASSERT(tensor->getRuntime()->isCuda());
        // The lifecycle of shapeInt64 is managed by the caller.
        auto shapeInt64 = make_ref<vector<int64_t>>();
        for (auto v : tensor->getDims())
            shapeInt64->push_back(v);
        DLTensor ret{
            .data = tensor->getRawDataPtr<void *>(),
            .device = DLDevice{.device_type = kDLCUDA, .device_id = 0},
            .ndim = (int32_t)shapeInt64->size(),
            .dtype =
                DLDataType{.code = (uint8_t)kDLFloat, .bits = 32, .lanes = 1},
            .shape = static_cast<int64_t *>(shapeInt64->data()),
            .strides = nullptr,
            .byte_offset = 0,
        };
        return {ret, shapeInt64};
    }

    pair<vector<TVMValue>, vector<int>>
    convertInOutToTVMArgs(const vector<DLTensorHolder> &inputs,
                          const DLTensorHolder &output) const {
        vector<TVMValue> values;
        vector<int> type_codes;

        // The order of inputs and outputs is consistant with definition of TVM
        // computation in Python, which is determined by AsTVMVisitor.
        values.emplace_back(TVMValue{.v_handle = (void *)&output.first});
        type_codes.emplace_back(kTVMDLTensorHandle);

        for (auto &in : inputs) {
            values.emplace_back(TVMValue{.v_handle = (void *)&in.first});
            type_codes.emplace_back(kTVMDLTensorHandle);
        }

        return {values, type_codes};
    }
};

REGISTER_KERNEL(Device::CUDA, OpType::MemBound, DataType::Float32,
                MemboundTVMPackedFunction,
                "Memobund_TVM_Ansor_packed_funciton");
}; // namespace infini

#endif
