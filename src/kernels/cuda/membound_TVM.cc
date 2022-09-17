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
        PerfRecord ret;
        auto op = as<MemBoundObj>(_op);
        auto context = dynamic_cast<const CudaRuntimeObj *>(_context);
        // TODO: invoke Ansor to tune a membound kernel
        std::string func = "mem_bound_" + std::to_string(op->getGuid());
        nnet::AsTVMVisitor visitor;
        visitor.dispatch(op->getNnetExpr());
        auto &&stmts = visitor.getStmts();
        auto &&inShapes = visitor.getInputShapes();
        auto &&outShape = visitor.getOutputShape();

        std::vector<std::string> inputs;
        std::vector<std::string> inputs;
        for (auto &&in : op->getInputs()) {
            inputs.emplace_back(getVarName(in));
        }
        std::string output = getVarName(op->getOutput());
        auto res = getAnsorCode(
            inShapes, std::vector<std::string>(inShapes.size(), "float32"),
            outShape, "float32", stmts, func, inputs, output);
        
        // Evaluate the kernel
        ret.time = timeit(
            [&]() {
                // TODO: run the kernel
            },
            [&]() { context->sync(); });
        return ret;
    }

    std::pair<std::string, std::string> getAnsorCode(
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
            auto temp = py::list(code[2]);
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
        return std::make_pair(funcCode, invokeCode);
    }
};

REGISTER_KERNEL(Device::CUDA, OpType::MemBound, DataType::Float32, MemboundTVM,
                "Memobund_TVM_Ansor");
}; // namespace infini