
#pragma once
#include <cstdio>

const int BATCH_SIZE = 32; // parallel tensor number.
const int DIM_MAX_SIZE = 8;

// Concat operator acts like element tensors composing to one big tensor,and
// split operator acts like one big tensor being composed by element
// tensors.
template <typename T> struct ElementTensorMetadata {
    T *data[BATCH_SIZE];
    int dimBgNo[BATCH_SIZE]; // the dimention begin no of the element tensor in
                             // the composed tensor.
    int dimSize[BATCH_SIZE]; // the dimention size of the element tensor.
    int nElements[BATCH_SIZE]; // the number of elements of the element tensor.
    void print() const {
        for (int i = 0; i < BATCH_SIZE; i++)
            printf("%d:(data=%p,dimBgNo=%d,dimSize=%d,nElements=%d)\n", i,
                   data[i], dimBgNo[i], dimSize[i], nElements[i]);
    }
};
template <typename T> struct ComposedTensorMetadata {
    int dimSize[DIM_MAX_SIZE];
    int stride[DIM_MAX_SIZE];
    T *data;
};

namespace infini {
void split_concat_kernel(const ElementTensorMetadata<float> &eleMeta,
                         const ComposedTensorMetadata<float> &compMeta, int dim,
                         int batchSize, int nDims, bool isSplit);
void split_concat_kernel(const ElementTensorMetadata<half> &eleMeta,
                         const ComposedTensorMetadata<half> &compMeta, int dim,
                         int batchSize, int nDims, bool isSplit);
} // namespace infini
