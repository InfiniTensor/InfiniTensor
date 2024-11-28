#include "utils.h"
namespace infini {
namespace infinimlir {
mlir::Type convertDataTypeToMlirType(mlir::MLIRContext *context,
                                     infini::DataType dtype) {
    if (dtype == DataType::Float32) {
        return mlir::FloatType::getF32(context);
    } else if (dtype == DataType::Float16) {
        return mlir::FloatType::getF16(context);
    } else if (dtype == DataType::Int32) {
        return mlir::IntegerType::get(context, 32, mlir::IntegerType::Signless);
    } else if (dtype == DataType::Int16) {
        return mlir::IntegerType::get(context, 16, mlir::IntegerType::Signless);
    } else if (dtype == DataType::Int8) {
        return mlir::IntegerType::get(context, 8, mlir::IntegerType::Signless);
    } else if (dtype == DataType::UInt8) {
        return mlir::IntegerType::get(context, 8, mlir::IntegerType::Unsigned);
    } else if (dtype == DataType::UInt16) {
        return mlir::IntegerType::get(context, 16, mlir::IntegerType::Unsigned);
    } else if (dtype == DataType::UInt32) {
        return mlir::IntegerType::get(context, 32, mlir::IntegerType::Unsigned);
    } else {
        throw std::runtime_error("Unsupported data type");
    }
}

DataType convertMlirTypeToDataType(mlir::Type type) {
    if (type.isF32()) {
        return DataType::Float32;
    } else if (type.isF16()) {
        return DataType::Float16;
    } else if (type.isSignlessInteger(8)) {
        return DataType::Int8;
    } else if (type.isSignlessInteger(16)) {
        return DataType::Int16;
    } else if (type.isSignlessInteger(32)) {
        return DataType::Int32;
    } else if (type.isUnsignedInteger(8)) {
        return DataType::UInt8;
    } else if (type.isUnsignedInteger(16)) {
        return DataType::UInt16;
    } else if (type.isUnsignedInteger(32)) {
        return DataType::UInt32;
    } else {
        throw std::runtime_error("Unsupported data type");
    }
}

std::vector<int64_t> int_to_int64t(const std::vector<int> &shape) {
    std::vector<int64_t> result(shape.size());
    std::transform(shape.begin(), shape.end(), result.begin(),
                   [](int dim) { return static_cast<int64_t>(dim); });
    return result;
}

std::vector<int> int64t_to_int(const std::vector<int64_t> &shape) {
    std::vector<int> result(shape.size());
    std::transform(shape.begin(), shape.end(), result.begin(),
                   [](int dim) { return static_cast<int>(dim); });
    return result;
}

} // namespace infinimlir
} // namespace infini