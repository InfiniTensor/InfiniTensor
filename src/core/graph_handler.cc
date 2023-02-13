#include "core/graph_handler.h"
#include "operators/element_wise.h"
#include "operators/matmul.h"
#include "operators/reshape.h"
#include "operators/unary.h"

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

// see operators/element_wise.h
#define DEFINE_ELEMENT_WISE_METHOD(name, obj)                                  \
    Tensor GraphHandlerObj::name(Tensor a, Tensor b, Tensor c) {               \
        if (c) {                                                               \
            g->addOpWithOutputs<obj##Obj>(a, b, c);                            \
            return c;                                                          \
        } else {                                                               \
            return g->addOp<obj##Obj>(a, b, c)->getOutput();                   \
        }                                                                      \
    }

DEFINE_ELEMENT_WISE_METHOD(add, Add)
DEFINE_ELEMENT_WISE_METHOD(sub, Sub)
DEFINE_ELEMENT_WISE_METHOD(mul, Mul)
DEFINE_ELEMENT_WISE_METHOD(div, Div)
DEFINE_ELEMENT_WISE_METHOD(pow, Pow)

// see operators/unary.h
#define DEFINE_UNARY_METHOD(name, obj)                                         \
    Tensor GraphHandlerObj::name(Tensor x, Tensor y) {                         \
        if (y) {                                                               \
            g->addOpWithOutputs<obj##Obj>(x, y);                               \
            return y;                                                          \
        } else {                                                               \
            return g->addOp<obj##Obj>(x, y)->getOutput();                      \
        }                                                                      \
    }

DEFINE_UNARY_METHOD(relu, Relu)
DEFINE_UNARY_METHOD(sigmoid, Sigmoid)
DEFINE_UNARY_METHOD(tanh, Tanh)
DEFINE_UNARY_METHOD(softmax, Softmax)
DEFINE_UNARY_METHOD(abs, Abs)
// see operators/reshape.h
DEFINE_UNARY_METHOD(identity, Identity)

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
