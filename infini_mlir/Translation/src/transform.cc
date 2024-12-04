#include "transform.h"
#include "ConvertToInfini.h"
#include "ConvertToMLIR.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Transforms/Passes.h"
#include "utils.h"

namespace infini {

namespace infinimlir {
Transformation::Transformation(mlir::MLIRContext &context)
    : context(context), pm(&context) {
    context.getOrLoadDialect<mlir::func::FuncDialect>();
    context.getOrLoadDialect<infini::infinimlir::InfiniDialect>();
}

Graph Transformation::transform(GraphObj *graph) {
    mlir::OpBuilder builder(&context);
    mlir::ModuleOp module = mlir::ModuleOp::create(builder.getUnknownLoc());

    std::unordered_map<UidBaseType, mlir::Value> tensor_value_map;
    std::vector<mlir::Type> inputTypes, outputTypes;
    for (const auto &tensor : graph->getInputs()) {
        inputTypes.push_back(mlir::RankedTensorType::get(
            int_to_int64t(tensor->getDims()),
            convertDataTypeToMlirType(&context, tensor->getDType())));
    }
    for (const auto &tensor : graph->getOutputs()) {
        outputTypes.push_back(mlir::RankedTensorType::get(
            int_to_int64t(tensor->getDims()),
            convertDataTypeToMlirType(&context, tensor->getDType())));
    }

    // Create the function
    auto func = mlir::func::FuncOp::create(
        builder.getUnknownLoc(), "main",
        builder.getFunctionType(inputTypes, outputTypes));

    // Create the entry block
    mlir::Block *entry_block = func.addEntryBlock();
    builder.setInsertionPointToStart(entry_block);

    for (size_t i = 0; i < graph->getInputs().size(); ++i) {
        mlir::Value arg = entry_block->getArgument(i);
        tensor_value_map[graph->getInputs()[i]->getFuid()] = arg;
    }
    for (auto const &tensor : graph->getWeights()) {
        auto tensor_type = mlir::RankedTensorType::get(
            int_to_int64t(tensor->getDims()),
            convertDataTypeToMlirType(&context, tensor->getDType()));
        auto tensor_op = builder.create<ConstantOp>(
            builder.getUnknownLoc(), tensor_type, tensor->size(),
            (uint64_t)tensor->getDataBlob()->getPtr<void *>());
        tensor_value_map[tensor->getFuid()] = tensor_op->getResult(0);
    }

    // transform the graph to mlir
    for (const auto &op : graph->getOperators()) {
        std::vector<mlir::Value> inputs, outputs;
        for (const auto &tensor : op->getInputs()) {
            if (tensor_value_map.find(tensor->getFuid()) !=
                tensor_value_map.end()) {
                inputs.push_back(tensor_value_map[tensor->getFuid()]);
            } else {
                std::cerr << "Error: tensor " << tensor->getFuid()
                          << " not found in tensor_value_map" << std::endl;
                exit(1);
            }
        }
        mlir::Operation *mlirOp = convertOpToMLIR(builder, op, inputs);
        for (int i = 0; i < op->numOutputs(); ++i) {
            mlir::Value output = mlirOp->getResult(i);
            tensor_value_map[op->getOutputs()[i]->getFuid()] = output;
        }
    }

    // create the return op
    std::vector<mlir::Value> return_values;
    for (const auto &tensor : graph->getOutputs()) {
        return_values.push_back(tensor_value_map[tensor->getFuid()]);
    }
    builder.create<mlir::func::ReturnOp>(builder.getUnknownLoc(),
                                         mlir::ValueRange(return_values));

    module.push_back(func);
    // Create the input tensors
    module.dump();
    if (mlir::failed(pm.run(module))) {
        module.emitError("Optimization failed.");
    }
    Graph new_graph = convertMLIRToInfini(module, graph->getRuntime());
    return new_graph;
}
} // namespace infinimlir
} // namespace infini
