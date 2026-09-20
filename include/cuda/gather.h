#pragma once
#include "core/data_type.h"
#include "core/operator.h"
#include "operators/gather.h"

namespace infini {
struct GatherMetaData {
    // Pointer to indices
    void *indexValue;
    // Type of index values
    DataType indexType;
    // Type of input and output data
    DataType dataType;
    // Axis of the gather operation
    int axis;
    // Rank of input
    int inNDim;
    // Rank of output
    int outNDim;
    // Rank of indices
    int idxNDim;
    // Shape of output
    int outDim[8];
    // Shape of indices
    int idxDim[8];
    // Strides of indices
    int idxStride[8];
    // Strides of input
    int inStride[8];
};

inline void initGatherMetaData(GatherMetaData &metaData,
                               const Ref<OperatorObj> &_op) {
    memset(&metaData, 0, sizeof(metaData));
    auto op = as<GatherBaseObj>(_op);
    Ref<TensorObj> in = op->getInputs(0);
    Ref<TensorObj> index = op->getInputs(1);
    Ref<TensorObj> out = op->getOutput();
    metaData.inNDim = in->getRank();
    metaData.outNDim = out->getRank();
    metaData.idxNDim = index->getRank();
    IT_ASSERT(metaData.inNDim >= 1 && metaData.inNDim <= 8 &&
              metaData.outNDim >= 0 && metaData.outNDim <= 8 &&
              metaData.idxNDim >= 0 && metaData.idxNDim <= 8,
              "CUDA Gather supports ranks up to 8");
    metaData.indexValue = index->getRawDataPtr<void *>();
    metaData.indexType = index->getDType();
    metaData.dataType = in->getDType();
    metaData.axis = op->getAxis();
    for (int i = 0; i < metaData.outNDim; ++i)
        metaData.outDim[i] = out->getDims()[i];
    for (int i = 0; i < metaData.idxNDim; ++i) {
        metaData.idxDim[i] = index->getDims()[i];
        metaData.idxStride[i] = index->getStride()[i];
    }
    for (int i = 0; i < metaData.inNDim; ++i) {
        metaData.inStride[i] = in->getStride()[i];
    }
}
template <typename T>
void gather_kernel(T *in, T *out, GatherMetaData metaData, size_t num);

void gather_elements_kernel(void *in, void *out, GatherMetaData metaData,
                            size_t num);
} // namespace infini
