#pragma once

const int MAX_DIM = 4;

// Pad operator acts like padding small(part) tensor into a big(whole) tensor.
// Slice operator acts like spling a big(whole) tensor into a small(part)
// tensor.
typedef struct {
    int begNum[MAX_DIM];     // pad or slice number at beginning
    int wholeNDim[MAX_DIM];  // dim size after padding or before slicing
    int partNDim[MAX_DIM];   // dim size before padding or after slicing
    int partStride[MAX_DIM]; // stride before padding or after slicing
} TransMetaData;

namespace infini {
void pad_slice_kernel(float *partData, float *wholeData,
                      const TransMetaData &metadata, int nDims, int num,
                      bool isPad);
} // namespace infini
