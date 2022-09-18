#include "core/kernel.h"
#include "cuda/cuda_runtime.h"
#include "operators/membound.h"
#include "operators/pooling.h"
#include "ffi/ffi_embed.h"
#include "nnet/Visitor/AsTVMVisitor.h"
#include "nvrtc.h"

namespace py = pybind11;

namespace infini {

class TVMRecord : public PerfRecord {
    // TODO: Add more attrs
public:
    size_t logSize, ptxSize;
    char *log, *ptx;

    TVMRecord(): logSize(0), ptxSize(0), log(nullptr), ptx(nullptr) { }
};

class MemboundTVM : public Kernel {
    void compute(const Operator &_op, const PerfRecord &record,
                 const RuntimeObj *_context) const override {
        auto op = as<MemBoundObj>(_op);
        // auto context = dynamic_cast<const CudaRuntimeObj *>(_context);
        // cudnnStatus_t stat;
        // void *const inData = (op->getInputs(0)->getRawDataPtr<void *>());
        // void *const outData = (op->getOutput()->getRawDataPtr<void *>());
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
        TVMRecord ret;
        auto op = as<MemBoundObj>(_op);
        auto context = dynamic_cast<const CudaRuntimeObj *>(_context);
        // TODO: invoke Ansor to tune a membound kernel
        std::string func = "mem_bound_" + std::to_string(op->getGuid());
        std::string kernelName = func + "kernel0";
        nnet::AsTVMVisitor visitor;
        visitor.dispatch(op->getNnetExpr());
        auto &&stmts = visitor.getStmts();
        auto &&inShapes = visitor.getInputShapes();
        auto &&outShape = visitor.getOutputShape();

        std::vector<std::string> inputs;
        for (auto &&in : op->getInputs()) {
            inputs.emplace_back(getVarName(in));
        }
        std::string output = getVarName(op->getOutput());
        auto res = getAnsorCode(
            inShapes, std::vector<std::string>(inShapes.size(), "float32"),
            outShape, "float32", stmts, func, inputs, output);
        
        // compile the kernel
        auto funcCode = res.first;
        // std::cout << funcCode << std::endl;
        std::cout << "get ansor code" << std::endl;
        auto invokeParams = res.second;
        std::string fileName = func + ".cu";
        nvrtcProgram prog;
        nvrtcCreateProgram(&prog,         // prog
                        funcCode.c_str(),         // buffer
                        fileName.c_str(),    // name
                        0,             // numHeaders
                        NULL,          // headers
                        NULL);         // includeNames
        const char *opts[] = {"--gpu-architecture=compute_80",
                      "--fmad=false"};
        nvrtcCompileProgram(prog,     // prog
                            2,        // numOptions
                            opts);    // options

        // copy ptx and log to ret
        size_t logSize;
        nvrtcGetProgramLogSize(prog, &logSize);
        char *log = new char[logSize];
        nvrtcGetProgramLog(prog, log);
        // Obtain PTX from the program.
        size_t ptxSize;
        nvrtcGetPTXSize(prog, &ptxSize);
        char *ptx = new char[ptxSize];
        nvrtcGetPTX(prog, ptx);
        std::cout << "compile and copy" << std::endl;
        // prepare for evaluation
        CUdevice cuDevice;
        CUcontext newContext;
        CUmodule module;
        CUfunction kernel;
        cuInit(0);
        cuDeviceGet(&cuDevice, 0);
        cuCtxCreate(&newContext, 0, cuDevice);
        cuModuleLoadDataEx(&module, ptx, 0, 0, 0);
        cuModuleGetFunction(&kernel, module, kernelName.c_str());
        std::vector<void *> args;
        args.push_back(op->getOutput()->getRawDataPtr<void *>());
        for (auto&& in : op->getInputs()) {
            args.push_back(in->getRawDataPtr<void *>());
        }
        std::vector<void *> argsPtr;
        for (auto &arg : args) {
            argsPtr.push_back(&arg);
        }
        std::cout << "prepare for evaluation" << std::endl;
        // Evaluate the kernel
        ret.time = timeit(
            [&]() {
                // TODO: run the kernel
                cuLaunchKernel(kernel,
                                invokeParams[0], invokeParams[1], invokeParams[2],
                                invokeParams[3], invokeParams[4], invokeParams[5],
                                0, NULL,
                                args.data(),
                                0);
                cuCtxSynchronize();
            },
            [&]() { context->sync(); });
        std::cout << "after evaluation, time: " << ret.time << std::endl;
        delete[]log;
        delete[]ptx;
        return ret;
    }

    std::pair<std::string, std::vector<int>> getAnsorCode(
        const std::vector<std::vector<int>> &inDims,
        const std::vector<std::string> &inDTypes, const std::vector<int> &outDims,
        const std::string &outDType, const std::string &lambda,
        const std::string &funcName, const std::vector<std::string> &inputNames,
        const std::string &outputName) const {
        std::string funcCode, invokeCode;
        std::vector<int> invokeParams;
        try {
            start_interpreter();
            auto func = py::module::import("cpp_plugin").attr("gen_ansor_op");
            py::tuple code = func(inDims, inDTypes, outDims, outDType, lambda,
                                funcName, inputNames, outputName);
            funcCode = py::str(code[0]), invokeCode = py::str(code[1]);
            std::cout << "return from python" << std::endl;
            std::cout << "funcCode: \n" << funcCode << "\n" << "invokeCode: \n" << invokeCode << std::endl;
            auto temp = py::list(code[3]);
            for (int i = 0; i < 6; ++i) {
                invokeParams.push_back(temp[i].cast<int>());
            }
            std::cout << "invoke params: \n";
            for (auto p : invokeParams) {
                std::cout << p << ", ";
            }
            std::cout << std::endl;
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
};

REGISTER_KERNEL(Device::CUDA, OpType::MemBound, DataType::Float32, MemboundTVM,
                "Memobund_TVM_Ansor");
}; // namespace infini