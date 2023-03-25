#include "core/tensor_base.h"
#include "core/blob.h"
#include "core/runtime.h"
namespace infini {

TensorBaseObj::TensorBaseObj(int dim, DataType dtype, Runtime runtime)
    : dim(dim), dtype(dtype), runtime(runtime) {}

}; // namespace infini
