#include "core/graph_handler.h"
#include "operators/matmul.h"

namespace infini {

static DataType dtype_repr_convert(int);

Tensor GraphHandlerObj::tensor(Shape dims, int dtype) {
    return g->addTensor(std::move(dims), dtype_repr_convert(dtype));
}

Tensor GraphHandlerObj::matmul(Tensor a, Tensor b, Tensor y, bool transA,
                               bool transB, Tensor bias, ActType act) {
    if (y) {
        g->addOpWithOutputs<MatmulObj>(std::move(a), std::move(b), y, transA,
                                       transB, std::move(bias), act);
        return y;
    } else {
        return g
            ->addOp<MatmulObj>(std::move(a), std::move(b), y, transA, transB,
                               std::move(bias), act)
            ->getOutput();
    }
}

static DataType dtype_repr_convert(int dtype) {
    switch ((OnnxDType)dtype) {
    case OnnxDType::FLOAT:
        return DataType::Float32;
    case OnnxDType::UINT32:
        return DataType::UInt32;
    default:
        IT_ASSERT(false, "Unsupported data type");
    }
}

} // namespace infini
