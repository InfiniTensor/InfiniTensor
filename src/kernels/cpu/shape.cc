#include "core/kernel.h"
#include "operators/unary.h"

namespace infini {

/// Writes out the dimensions of the input as the int64 list ONNX asks for.
///
/// The dimensions are already known once shapes have been inferred, so there is
/// nothing to work out here beyond handing them over in the layout the rest of
/// the graph reads them from.
class NaiveShape : public CpuKernelWithoutConfig {
    void compute(const Operator &_op,
                 const RuntimeObj *context) const override {
        const auto &dims = _op->getInputs(0)->getDims();
        auto outPtr = _op->getOutput()->getRawDataPtr<int64_t *>();
        for (size_t i = 0; i < dims.size(); ++i) {
            outPtr[i] = static_cast<int64_t>(dims[i]);
        }
    }
};

REGISTER_KERNEL(Device::CPU, OpType::Shape, NaiveShape, "ShapeNaive_CPU");

} // namespace infini
