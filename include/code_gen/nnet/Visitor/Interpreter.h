#pragma once
#include "code_gen/nnet/visitor.h"

namespace nnet {

class Interpreter : public Functor<int()> {
  public:
    using ttype = int; // Test data type
    using rtype = int; // Return data type
    using Position = vector<int>;
    using Inputs = unordered_map<string, Ref<vector<ttype>>>;
    using Iteration = PtrUmap<Var, int>;

  private:
    // cache the input value
    Inputs inputs;
    vector<Iteration> iterations;
    vector<Position> positions;

    rtype visit_(const Constant &c) override;
    rtype visit_(const BinaryOp &c) override;
    rtype visit_(const RangeOp &c) override;
    rtype visit_(const Subscript &c) override;
    rtype visit_(const Var &c) override;
    rtype visit_(const Tensor &c) override;
    // int visit_(const Func &c); // Future work

    static Inputs genInputStartingFromZero(const RangeOp &range);

  public:
    Interpreter(Inputs _inputs, int _verbose = 0)
        : Functor(_verbose), inputs(_inputs) {}
    Interpreter(RangeOp range, int _verbose = 0);

    /**
     * @brief Calculate the output at specified poistions
     *
     * @param expr The expression to be calculated.
     * @param poses Positions of output.
     * @return vector<int> Value of output.
     */
    vector<rtype> interpret(const Expr &expr, const vector<Position> &poses);
    /**
     * @brief Calculate the output at equally spaced positions
     *
     * @param expr The expression to be calculated.
     * @param nPoses The number of calculated output positions.
     * @return vector<int> Value of output.
     */
    vector<rtype> interpretUniformSample(const RangeOp &range,
                                         int nPoses = 100);
    vector<rtype> interpretAllOutput(const RangeOp &range);
};

} // namespace nnet