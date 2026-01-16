#ifndef NINETOOTHED_H
#define NINETOOTHED_H

#include <stdint.h>

enum NineToothedDataType {
    NINETOOTHED_INT8,
    NINETOOTHED_INT16,
    NINETOOTHED_INT32,
    NINETOOTHED_INT64,
    NINETOOTHED_UINT8,
    NINETOOTHED_UINT16,
    NINETOOTHED_UINT32,
    NINETOOTHED_UINT64,
    NINETOOTHED_FLOAT16,
    NINETOOTHED_BFLOAT16,
    NINETOOTHED_FLOAT32,
    NINETOOTHED_FLOAT64
};

typedef struct {
    void *data;
    uint64_t *shape;
    int64_t *strides;
} NineToothedTensor;

typedef void *NineToothedStream;

typedef int NineToothedResult;

#endif // NINETOOTHED_H
