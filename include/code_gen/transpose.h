#ifndef TRANSPOSE_H
#define TRANSPOSE_H

#include <string>
#include <vector>

namespace tpm {

class DecisionTree {
  private:
    std::vector<std::string> condition; // for leaf node condition is expression
    DecisionTree *thenNode;
    DecisionTree *elseNode;

  public:
    DecisionTree(std::vector<std::string> condition,
                 DecisionTree *thenNode = nullptr,
                 DecisionTree *elseNode = nullptr)
        : condition(condition), thenNode(thenNode), elseNode(elseNode) {}
    std::vector<std::string> getCond() { return condition; }
    void putCond(std::vector<std::string> &_cond) { condition = _cond; }
    DecisionTree *getThenNode() { return thenNode; }
    DecisionTree *getElseNode() { return elseNode; }
};

class TransBasic {
  public:
    virtual ~TransBasic() {}
    virtual DecisionTree *putTranspose(DecisionTree *, std::vector<int> &) = 0;

    virtual void print() = 0;

    virtual void getOptypeDim(std::string &optype, std::vector<int> &dim,
                              std::vector<int> &extra) = 0;
};

class TransSplit : public TransBasic {
    // (..., dims[index], ...) -> (..., dims[index]/in_dims, in_dims, ...)
  private:
    int index, in_dim, extr, copys;

  public:
    TransSplit(int index, int in_dim, int extr = 0, int copys = 0)
        : index(index), in_dim(in_dim), extr(extr), copys(copys) {}
    DecisionTree *putTranspose(DecisionTree *, std::vector<int> &);
    void dfsSplit(DecisionTree *tree, std::string newvar);
    void print();
    void getOptypeDim(std::string &optype, std::vector<int> &dim,
                      std::vector<int> &extra);
};

class TransFuse : public TransBasic {
    // (..., dims[index], dims[index+1], ...) ->
    // (..., dims[index]*dims[index+1], ...)
  private:
    int index, pad, dels;

  public:
    TransFuse(int index, int pad = 0, int dels = 0)
        : index(index), pad(pad), dels(dels) {}
    DecisionTree *putTranspose(DecisionTree *, std::vector<int> &);
    void dfsFuse(DecisionTree *tree, std::string newvar1, std::string newvar2);
    void print();
    void getOptypeDim(std::string &optype, std::vector<int> &dim,
                      std::vector<int> &extra);
};

class TransReorder : public TransBasic {
  private:
    std::vector<int> permu;

  public:
    TransReorder(std::vector<int> permu) : permu(permu) {}
    DecisionTree *putTranspose(DecisionTree *, std::vector<int> &);
    void dfsReorder(DecisionTree *, std::vector<int> &);
    void print();
    void getOptypeDim(std::string &optype, std::vector<int> &dim,
                      std::vector<int> &extra);
};

class TransposeEngine { // Singleton Pattern
  private:
    TransposeEngine() {}
    ~TransposeEngine() {}
    TransposeEngine(const TransposeEngine &) = delete;
    TransposeEngine &operator=(const TransposeEngine &) = delete;

  private:
    std::string getLambdaExp(DecisionTree *);

  public:
    static TransposeEngine &getInstance() {
        static TransposeEngine instance;
        return instance;
    }

    std::string getLambda(std::vector<TransBasic *> oper,
                          const std::vector<int> &dims);
};

} // end of namespace tpm

#endif
