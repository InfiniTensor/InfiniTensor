#include "ConvertToInfini.h"
#include "utils.h"

namespace infini {

namespace infinimlir {

void handleAddOp(Graph &g, mlir::Operation *op,
                 llvm::DenseMap<mlir::Value, Tensor> &tensorMap) {
    auto addOp = llvm::cast<infinimlir::AddOp>(op);

    // create op inputs Tensor
    std::vector<Tensor> inputs;
    for (unsigned i = 0; i < addOp.getNumOperands(); ++i) {
        mlir::Value value = addOp.getOperand(i);
        if (tensorMap.find(value) == tensorMap.end()) {
            auto shape =
                mlir::cast<mlir::RankedTensorType>(value.getType()).getShape();
            Tensor new_tensor = g->addTensor(
                int64t_to_int(shape),
                convertMlirTypeToDataType(
                    mlir::cast<mlir::RankedTensorType>(value.getType())
                        .getElementType()));
            tensorMap[value] = new_tensor;
            inputs.push_back(new_tensor);
        } else {
            inputs.push_back(tensorMap[value]);
        }
    }

    // create op output Tensor
    mlir::Value output = addOp.getResult();
    auto shape =
        mlir::cast<mlir::RankedTensorType>(output.getType()).getShape();
    Tensor output_tensor =
        g->addTensor(int64t_to_int(shape),
                     convertMlirTypeToDataType(
                         mlir::cast<mlir::RankedTensorType>(output.getType())
                             .getElementType()));
    tensorMap[output] = output_tensor;
    // create op
    g->addOpWithOutputs<AddObj>(inputs[0], inputs[1], output_tensor);
}

void handleConstantOp(Graph &g, mlir::Operation *op,
                      llvm::DenseMap<mlir::Value, Tensor> &tensorMap) {
    auto constantOp = llvm::cast<infinimlir::ConstantOp>(op);

    // create op output Tensor
    mlir::Value output = constantOp.getResult();
    auto shape =
        mlir::cast<mlir::RankedTensorType>(output.getType()).getShape();
    Tensor output_tensor =
        g->addTensor(int64t_to_int(shape),
                     convertMlirTypeToDataType(
                         mlir::cast<mlir::RankedTensorType>(output.getType())
                             .getElementType()));
    void *data_ptr = (void *)(uintptr_t)constantOp.getDataPtr();
    output_tensor->setDataBlob(make_ref<BlobObj>(g->getRuntime(), data_ptr));
    output_tensor->setWeight();
    tensorMap[output] = output_tensor;
}

Graph convertMLIRToInfini(mlir::ModuleOp module, Runtime runtime) {
    llvm::DenseMap<mlir::Value, Tensor> tensorMap;
    Graph g = make_ref<GraphObj>(runtime);
    for (auto func : module.getOps<mlir::func::FuncOp>()) {
        for (auto &block : func.getBlocks()) {
            for (auto &op : block.getOperations()) {
                // op.dump();
                if (llvm::isa<infinimlir::AddOp>(op)) {
                    handleAddOp(g, &op, tensorMap);
                } else if (llvm::isa<infinimlir::ConstantOp>(op)) {
                    handleConstantOp(g, &op, tensorMap);
                } else if (llvm::isa<mlir::func::ReturnOp>(op)) {
                    continue;
                } else {
                    throw std::runtime_error("Unsupported op");
                }
            }
        }
    }
    return g;
}

} // namespace infinimlir

} // namespace infini
