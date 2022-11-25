#include "operators/membound.h"
#include "core/kernel.h"
#include "nnet/Visitor/Interpreter.h"
#include <cstring>

namespace infini {

class MemboundInterpreter : public Kernel {
    void compute(const Operator &_op, const PerfRecord &record,
                 const RuntimeObj *_context) const override {
        auto op = as<MemBoundObj>(_op);
        auto output = op->getOutput();
        output->dataMalloc();
        // TODO: use uint32_t in Interpreter
        std::unordered_map<std::string,
                           nnet::Ref<std::vector<nnet::Interpreter::ttype>>>
            rangeInputs;
        // TODO: avoid this copy by modifying Interpreter
        for (int i = 0; i < op->numInputs(); i++) {
            auto input = op->getInputs(i);
            auto data = nnet::make_ref<std::vector<nnet::Interpreter::ttype>>(
                input->getBytes() / sizeof(nnet::Interpreter::ttype));
            memcpy(data->data(), op->getInputs(i)->getRawDataPtr<void *>(),
                   input->getBytes());
            rangeInputs.insert({op->getNnetInputs()[i]->getName(), data});
        }
        // for (size_t i = 0; i < op->getInputs().size(); ++i) {
        //     rangeInputs.insert({op->getNnetInputs()[i]->getName(),
        //                         op->getInputs(i)->getDataBlob()});
        // }

        nnet::RangeOp range = nnet::as<nnet::RangeOpNode>(op->getNnetExpr());
        // const auto &rangeShape = range->getOutputShape();
        // const auto &outputShape = output->getDims();
        // rangeShape and outputShape may extra dims of length 1.
        // But their sizes should be the same.
        IT_ASSERT((ssize_t)range->getOutputSize() == (ssize_t)output->size());
        // const ssize_t iEnd = range->getOutputSize();
        // #pragma omp parallel for default(none)  shared(range, output,
        // rangeShape, outputShape, rangeInputs, iEnd)
        //         for (ssize_t i = 0; i < iEnd; i++) {
        //             std::vector<int> rangePos(range->getNumOutputDims(), 0);
        //             std::vector<int> outputPos(outputShape.size(), 0);
        //             ssize_t t = i;
        //             for (int j = range->getNumOutputDims() - 1; 0 <= j; j--)
        //             {
        //                 int extent = rangeShape[j];
        //                 rangePos[j] = t % extent;
        //                 t /= extent;
        //             }
        //             t = i;
        //             for (int j = outputShape.size() - 1; 0 <= j; j--) {
        //                 int extent = outputShape[j];
        //                 outputPos[j] = t % extent;
        //                 t /= extent;
        //             }
        //             auto vals =
        //                 nnet::Interpreter(rangeInputs).interpret(range,
        //                 {rangePos});
        //             output->setData(outputPos, vals[0]);
        //         }
        auto vals = nnet::Interpreter(rangeInputs).interpretAllOutput(range);
        // output->setData(outputPos, vals[0]);
        vector<uint32_t> valsUint(vals.size());
        for (size_t i = 0; i < vals.size(); ++i)
            valsUint[i] = (uint32_t)vals[i];
        output->copyin(valsUint);
    }

    void compute(const Operator &op, const RuntimeObj *context) const override {
        compute(op, {}, context);
    }

    PerfRecord tune(const Operator &op,
                    const RuntimeObj *context) const override {
        return make_ref<PerfRecordObj>(
            timeit([&]() { compute(op, context); }, []() {}, 0, 1));
    }
};

REGISTER_KERNEL(Device::CPU, OpType::MemBound, DataType::UInt32,
                MemboundInterpreter, "MemboundInterpreter_CPU");

} // namespace infini
