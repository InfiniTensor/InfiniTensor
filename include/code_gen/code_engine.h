#pragma once
#include "graph.h"
#include "operator.h"
#include "tensor.h"
#include <string>
#include <unordered_map>
#include <vector>

namespace tpm {

class CodeEngine {
  private:
    int tabs = 0;
    std::string head, main;
    std::shared_ptr<PerfEngine> perfEngine;
    std::unordered_map<const TransposeOp *,
                       std::shared_ptr<std::vector<const TransposeOp *>>>
        transposeMap;

  public:
    CodeEngine() {}
    ~CodeEngine() {}

    // CUDNN helper
    std::string actToStr(Operator::ActType act);

    // Code fragment
    std::string getVarName(const Tensor &t);
    std::string getTensorDescName(const Tensor &t);
    std::string getFilterDescName(const Tensor &t);
    std::string getDescName(const Operator &op);
    void genHeader();

    // Dispatcher
    void genDesc(const Operator &op);
    void genCompute(const Operator &op);

    // Tensors
    Dim getDim(const Tensor &t);
    size_t getTensorNElem(const Tensor &t);
    size_t getTensorSize(const Tensor &t);
    void genTensorAlloc(const Tensor &t, bool isConvBias = false);
    void genTensorFree(const Tensor &t);

    // Desc generator
    void genConvDesc(const ConvOp &op);
    void genMatmulDesc(const MatmulOp &op);
    void genPadDesc(const PadOp &op);
    void genSliceDesc(const SliceOp &op);
    void genActivationDesc(const ActivationOp &op);
    void genAvgPoolDesc(const AvgPoolOp &op);
    void genMaxPoolDesc(const MaxPoolOp &op);
    void genAddDesc(const AddOp &op);
    void genMulDesc(const MulOp &op);
    void genTransposeDesc(const TransposeOp &op);
    void genGatherDesc(const GatherOp &op);
    void genSplitDesc(const SplitOp &op);
    void genConcatDesc(const ConcatOp &op);
    void genExtendDesc(const ExtendOp &op);
    void genReshapeDesc(const ReshapeOp &op);
    void genSoftmaxDesc(const SoftmaxOp &op);
    void genMemBoundDesc(const MemBoundOp &op);
    void genConvTransDesc(const ConvTransOp &op);
    void genG2BMMDesc(const G2BMMOp &op);
    void genGBMMLDesc(const GBMMLOp &op);
    void genBatchNormDesc(const BatchNormOp &op);

    // Compute generator
    void genConvCompute(const ConvOp &op);
    void genMatmulCompute(const MatmulOp &op);
    void genPadCompute(const PadOp &op);
    void genSliceCompute(const SliceOp &op);
    void genActivationCompute(const ActivationOp &op);
    void genPoolCompute(const Operator &op); // shared for Max and Avg
    void genAddCompute(const AddOp &op);
    void genMulCompute(const MulOp &op);
    void genTransposeCompute(const TransposeOp &op);
    void genGatherCompute(const GatherOp &op);
    void genSplitCompute(const SplitOp &op);
    void genConcatCompute(const ConcatOp &op);
    void genExtendCompute(const ExtendOp &op);
    void genReshapeCompute(const ReshapeOp &op);
    void genSoftmaxCompute(const SoftmaxOp &op);
    void genMemBoundCompute(const MemBoundOp &op);
    void genConvTransCompute(const ConvTransOp &op);
    void genG2BMMCompute(const G2BMMOp &op);
    void genGBMMLCompute(const GBMMLOp &op);
    void genReduce_merge_conv_3x3_1x1(const MemBoundOp &op);
    void genBatchNormCompute(const BatchNormOp &op);

    // Code tools
    int clear();
    int shiftTab(int n);
    int emit(std::string line);
    std::string render();

    // TVM
    std::pair<std::string, std::string>
    getTVMCode(const std::vector<std::vector<int>> &inDims,
               const std::vector<std::string> &inDTypes,
               const std::vector<int> &outDims, const std::string &lambda,
               const std::string &funcName,
               const std::vector<std::string> &inputNames,
               const std::string &outputName);
    std::pair<std::string, std::string>
    getAnsorCode(const std::vector<std::vector<int>> &inDims,
                 const std::vector<std::string> &inDTypes,
                 const std::vector<int> &outDims, const std::string &outDType,
                 const std::string &lambda, const std::string &funcName,
                 const std::vector<std::string> &inputNames,
                 const std::string &outputName);

    // File
    inline bool check_existed(const std::string &name);

    // Gen code
    std::string genCode(std::shared_ptr<SubGraph> &graph);
    int genCode(std::shared_ptr<SubGraph> &graph, const std::string &filename);

    std::pair<std::string, std::string>
    genTranspose(const std::vector<const TransposeOp *> ops,
                 const std::string &funcName, const std::string &inputName,
                 const std::string &outputName);

    void importPerfEngine(std::shared_ptr<PerfEngine> perfEngine_);
};

} // namespace tpm
