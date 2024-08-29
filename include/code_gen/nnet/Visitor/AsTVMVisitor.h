#pragma once
#include "code_gen/nnet/Visitor/StrideVisitor.h"
#include "code_gen/nnet/visitor.h"

namespace nnet {

class AsTVMVisitor : public Functor<std::string(void)> {
  private:
    int nStage = 0, curStage = -1;
    std::unordered_map<std::string, int> offset;
    std::vector<std::string> inputs;
    std::string output;
    std::vector<std::string> pythonVars;
    std::vector<std::vector<int>> inputShapes;
    std::vector<int> outputShape;
    std::string stmts;

  public:
    std::string getStmts() const;

    const std::vector<std::string> &getInputs() const { return inputs; }
    const std::string &getOutput() const { return output; }

    const std::vector<std::vector<int>> &getInputShapes() const {
        return inputShapes;
    }
    const std::vector<int> &getOutputShape() const { return outputShape; }

    std::string visit_(const Constant &c) override;
    std::string visit_(const BinaryOp &c) override;
    std::string visit_(const Func &c) override;
    std::string visit_(const RangeOp &c) override;
    std::string visit_(const Subscript &c) override;
    std::string visit_(const Var &c) override;
    std::string visit_(const Tensor &c) override;
};

} // namespace nnet