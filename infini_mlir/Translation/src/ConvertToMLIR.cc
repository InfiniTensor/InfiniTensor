#include "ConvertToMLIR.h"
#include "utils.h"

namespace infini {

namespace infinimlir {

std::unique_ptr<OperationConverter> createConverter(OpType type) {
    if (type == OpType::Add) {
        return std::make_unique<AddConverter>();
    } else {
        return nullptr;
    }
}

mlir::Operation *convertOpToMLIR(mlir::OpBuilder &builder, const Operator &op,
                                 const std::vector<mlir::Value> &inputs) {
    auto converter = createConverter(op->getOpType());
    if (converter) {
        return converter->convertToMLIR(builder, op, inputs);
    } else {
        std::cerr << "Unsupported operator type: " << op->getOpType().toString()
                  << std::endl;
        return nullptr;
    }
}

mlir::Operation *
AddConverter::convertToMLIR(mlir::OpBuilder &builder, const Operator &op,
                            const std::vector<mlir::Value> &inputs) {
    IT_ASSERT(inputs.size() == 2);
    auto input1 = inputs[0];
    auto input2 = inputs[1];
    auto resultType = mlir::RankedTensorType::get(
        int_to_int64t(op->getOutput()->getDims()),
        convertDataTypeToMlirType(builder.getContext(),
                                  op->getOutput()->getDType()));
    return builder.create<AddOp>(builder.getUnknownLoc(), resultType, input1,
                                 input2);
}

} // namespace infinimlir

} // namespace infini