#pragma once
#include "code_gen/nnet/derivator.h"

namespace nnet {

class Pass {
  private:
    VecExpr transformations;

  protected:
    Derivator &derivator;
    string passName;
    /**
     * @brief // False if does not add log in Derivator. It should be false for
     * single Pass test to avoid mismatch of passInfos and passMsgs  due to
     * different number of "run" and "nextStep".
     */
    bool enableLogging, enableDebug;

    virtual void transform(Formula &origin, int depth, Expr &rCur) = 0;
    void nextStep(Formula &origin, int depth, Expr &rCur, Expr newCur,
                  const string &ruleInfo = "");

    Var getNewVar();
    string newTensorName();

  private:
    void initialize(Formula &origin, const Expr &rCur);
    void finalize();

  public:
    Pass(Derivator &derivator, const string &passName);
    virtual ~Pass();

    void run(Formula &origin, int dfsDepth, Expr &rCur);
    void setEnableLogging(bool value);
    void setEnableDebug(bool value);
    const VecExpr &getTransformations();
};

} // namespace nnet