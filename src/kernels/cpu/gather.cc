#include "operators/gather.h"
#include "core/kernel.h"
#include <cstring>

namespace infini {

class GatherCpu final : public CpuKernelWithoutConfig {
    template <typename IndexT> void doCompute(const Operator &_op) const {
        auto op = as<GatherObj>(_op);
        IT_ASSERT(op != nullptr);

        const Tensor data = op->getInputs(0);
        const Tensor indices = op->getInputs(1);
        const Tensor output = op->getOutput();
        const Shape dataShape = data->getDims();
        const int axis = op->getAxis();

        size_t outer = 1;
        for (int i = 0; i < axis; ++i) {
            outer *= static_cast<size_t>(dataShape[i]);
        }

        size_t inner = 1;
        for (size_t i = static_cast<size_t>(axis + 1); i < dataShape.size();
             ++i) {
            inner *= static_cast<size_t>(dataShape[i]);
        }
        const int64_t axisDim = dataShape[axis];
        const size_t indices_count = indices->size();
        const size_t elementBytes = data->getDType().getSize();
        const size_t copyBytes = inner * elementBytes;

        const auto *indexData = indices->getRawDataPtr<const IndexT *>();
        const auto *input = data->getRawDataPtr<const uint8_t *>();
        auto *result = output->getRawDataPtr<uint8_t *>();

        for (size_t outerIndex = 0; outerIndex < outer; ++outerIndex) {
            for (size_t indexOffset = 0; indexOffset < indices_count;
                 ++indexOffset) {
                int64_t index = static_cast<int64_t>(indexData[indexOffset]);

                IT_ASSERT(index >= -axisDim && index < axisDim,
                          "Gather index is out of range");

                if (index < 0)
                    index += axisDim;

                const size_t inputElementOffset =
                    (outerIndex * static_cast<size_t>(axisDim) +
                     static_cast<size_t>(index)) *
                    inner;
                const size_t outputElementOffset =
                    (outerIndex * indices_count + indexOffset) * inner;
                std::memcpy(result + outputElementOffset * elementBytes,
                            input + inputElementOffset * elementBytes,
                            copyBytes);
            }
        }
    }

    void compute(const Operator &_op,
                 const RuntimeObj *context) const override {
        const DataType indicesType = _op->getInputs(1)->getDType();

        if (indicesType == DataType::Int32)
            doCompute<int32_t>(_op);
        else if (indicesType == DataType::Int64)
            doCompute<int64_t>(_op);
        else
            IT_ASSERT(false, "Gather indices type Wrong!");
    }
};

REGISTER_KERNEL(Device::CPU, OpType::Gather, GatherCpu, "Gather_CPU");

} // namespace infini