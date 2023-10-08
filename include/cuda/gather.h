#pragma once
#include "core/data_type.h"
#include "core/operator.h"
#include "operators/gather.h"

namespace infini {
struct GatherMetaData {
    void *indexValue;
    DataType indexType;
    DataType dataType;
    int axis;
    int inNDim;
    int outNDim;
    int idxNDim;
    int outDim[4];
    int idxDim[4];
    int idxStride[4];
    int inStride[4];
};

inline void initGatherMetaData(GatherMetaData &metaData, const Ref<OperatorObj> &_op) {
    memset(&metaData, 0, sizeof(metaData));
    auto op = as<GatherBaseObj>(_op);
    Ref<TensorObj> in = op->getInputs(0);
    Ref<TensorObj> index = op->getInputs(1);
    Ref<TensorObj> out = op->getOutput();
    metaData.indexValue = index->getRawDataPtr<void *>();
    metaData.indexType = index->getDType();
    metaData.dataType = in->getDType();

    printf("%lu\n", sizeof(metaData.indexType));

    metaData.axis = op->getAxis();
    metaData.inNDim = in->getRank();
    metaData.outNDim = out->getRank();
    metaData.idxNDim = index->getRank();
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
void gather_kernel(float *in, float *out, GatherMetaData metaData, size_t num);

void gather_elements_kernel(void *in, void *out, GatherMetaData metaData,
                            size_t num);
} // namespace infini
