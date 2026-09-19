#include "operators/slice.h"
#include "core/kernel.h"
#include "utils/operator_utils.h"
#include <algorithm>

namespace infini {

class NaiveSlice : public CpuKernelWithoutConfig {
    template <typename T>
    void doCompute(const Operator &_op, const RuntimeObj *context) const {
        auto op = as<SliceObj>(_op);
        const auto &input = op->getInputs(0);
        const auto &output = op->getOutput();
        // No element is addressed for an empty window. In particular, the
        // contiguous-run calculation below must not divide by a zero run.
        if (output->size() == 0)
            return;
        const T *inPtr = input->getRawDataPtr<T *>();
        T *outPtr = output->getRawDataPtr<T *>();

        // One range per input dimension, whether the model named that axis or
        // not: the operator fills in the whole extent for the ones it was not
        // asked about, so there is nothing to line up here.
        const Shape starts = op->getStarts();
        const Shape steps = op->getSteps();
        const Shape &outDims = output->getDims();
        const Shape &inDims = input->getDims();
        IT_ASSERT(starts.size() == inDims.size());
        IT_ASSERT(steps.size() == inDims.size());
        IT_ASSERT(outDims.size() == inDims.size());
        for (const auto step : steps) {
            // This kernel's contiguous-run offsets support forward steps.
            // Shape inference also describes reverse ranges, but their CPU
            // execution still needs a separate implementation.
            IT_ASSERT(step > 0, "a backwards slice is not supported yet");
        }

        Shape inStride(inDims.size(), 1);
        for (auto i = inDims.size(); i > 1; --i) {
            inStride[i - 2] = inStride[i - 1] * inDims[i - 1];
        }

        // How many elements at the end of the output are adjacent in the input
        // as well, so that they can be copied together rather than one strided
        // element at a time. A dimension taken whole with a step of one leaves
        // its elements adjacent; and the outermost such dimension may also be a
        // narrowed window, because a window of adjacent runs is itself one run.
        // This is the ordinary case: dropping the tail of a shape vector, or
        // taking a range of rows.
        size_t run = 1;
        size_t walked = inDims.size();
        while (walked > 0) {
            const size_t axis = walked - 1;
            if (steps[axis] != 1 || outDims[axis] != inDims[axis]) {
                break;
            }
            run *= static_cast<size_t>(outDims[axis]);
            --walked;
        }
        if (walked > 0 && steps[walked - 1] == 1) {
            run *= static_cast<size_t>(outDims[walked - 1]);
            --walked;
        }

        // Every dimension past `walked` contributes only where its range
        // begins, since the run covers the rest of it.
        size_t fixedOffset = 0;
        for (size_t axis = walked; axis < inDims.size(); ++axis) {
            fixedOffset += static_cast<size_t>(starts[axis]) *
                           static_cast<size_t>(inStride[axis]);
        }

        const Shape outer(outDims.begin(), outDims.begin() + walked);
        const size_t groups = output->size() / run;
        for (size_t g = 0; g < groups; ++g) {
            // Where this run sits among the dimensions that are walked rather
            // than copied.
            const Shape position = locate_index(g, outer);
            size_t offset = fixedOffset;
            for (size_t axis = 0; axis < walked; ++axis) {
                offset += static_cast<size_t>(starts[axis] +
                                              position[axis] * steps[axis]) *
                          static_cast<size_t>(inStride[axis]);
            }
            std::copy_n(inPtr + offset, run, outPtr + g * run);
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
            CASE(6); // DataType::Int32
            break;
            CASE(7); // DataType::Int64
            break;
            CASE(11); // DataType::Double
            break;
            CASE(12); // DataType::UInt32
            break;
        default:
            IT_TODO_HALT();
        }
    }
};

REGISTER_KERNEL(Device::CPU, OpType::Slice, NaiveSlice, "SliceNaive_CPU");

} // namespace infini
