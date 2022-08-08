#pragma once
#include "common.h"
#include "expr.h"
#include <iostream>

namespace nnet {

using PatternTensorMap = vector<Tensor>;
using PatternIterRangeMap = PtrMap<Iterator, VarRangePair>;

enum class MismatchType {
    // Search required (undetermined)
    MoreVar,
    LessVar,
    StrideMismatch,
    // guided DLT (determined)
    DLMismatch,
    OutputDLMismatch,
    OutputDimismatch
};
struct Mismatch {
    MismatchType type;
    int bitmap; // Row ID of IT
    PtrMap<Iterator, Iterator>
        mappingIter_r; // For DLT mismatch, iters are mapped
    Mismatch(MismatchType _type, int _bitmap) : type(_type), bitmap(_bitmap) {}
    Mismatch(MismatchType _type, int _bitmap,
             PtrMap<Iterator, Iterator> _mappingIter_r)
        : type(_type), bitmap(_bitmap), mappingIter_r(_mappingIter_r) {}
};
class Pattern;
class IteratorTable {
  protected:
    //     using Appearance = map<string, vector<pair<Tensor, int>>>;
    // using StrideTable = map<TensorNode *, vector<tuple<string, int, int>>>;
    // // Var, dim, stride
    RangeOp rangeOp;
    // To real tensor
    // FIXME: redundent
    Appearance appearance;
    vector<Tensor> tensors;       // original tensor sequence
    vector<Subscript> subscripts; // original subscripts sequence
    StrideTable strideTable;      // TODO [Refactor]: rename strideTable
    PatternIterRangeMap iterToRange;

    // mapping
    vector<int> tensorMap; // [index for tensors] -> tensorID in pattern
    PtrMap<Iterator, Iterator> iterMap; // [expr iter] -> pattern iter

    // final data
    vector<vector<Iterator>> posTable; // [Tensor bitmap]=[Iterator]
    vector<vector<vector<Iterator>>>
        iterInTensorDim; // [tensorID][dimOfTensor]=[Iterator],
                         // stride in each dim may be add

    vector<vector<PtrMap<Iterator, int>>>
        strideInDim; // [tensorID][dimOfTensor][Iterator]=stride,
                     // stride in each dim may be add

    PtrMap<Iterator, vector<int>> strideInTensor; // [Iterator][tensorID]=stride

    // final data: auxiliary data
    vector<int> tensorIDMap_r;
    PatternTensorMap tensorMap_r;
    PatternIterRangeMap iterToRange_r;

  public:
    virtual ~IteratorTable() {}
    IteratorTable() {}
    IteratorTable(const IteratorTable &) = delete;
    [[nodiscard]] bool analyzeExpr(const RangeOp &rangeOp);
    // mapTensors
    void buildTable(const vector<int> &_tensorMap);
    void buildTableWithDefaultMap();
    /**
     * @brief Check whether the expression match a pattern. If not, return the
     * detailed reason for guided search.
     *
     * @param patternIT
     * @return vector<int> mismatched IT rows/tensors for guided DLT.
     */
    vector<Mismatch> matchPatternIT(const Pattern &patternIT);
    void matchIterators();
    int getNumInputs() const { return tensors.size(); }
    int getNumTensors() const { return tensors.size() + 1; }
    int getNumRows() const { return 1 << getNumTensors(); }
    int getNumIterators() const { return strideTable.size(); }
    // vector<Tensor> tensorMap_r(
    //     pattern.nInputs); // [pattern tensor ID] -> real tensor
    // map<string, VarRangePair> iterToRange_r; // [pattern iter] -> iter &
    // range
    auto getTables() const {
        return tuple(posTable, iterInTensorDim, strideInTensor);
    }
    const auto &getStrideInDim() const { return strideInDim; }
    vector<vector<Iterator>> getIterInTensorDim(int tensorID) const {
        return iterInTensorDim[tensorID];
    }
    const vector<Iterator> &getPosTable(int bitmap) const {
        return posTable[bitmap];
    }
    pair<PatternTensorMap, PatternIterRangeMap> getReverseMap() const;

    int getStridesInTensor(Iterator iter, int tensorID) const;
    vector<int> getIterDimInTensor(int tensorID, const Iterator &iter) const;
    Tensor getTensor(int tensorID) const { return tensorMap_r[tensorID]; }
    Subscript getSubscript(int tensorID) const {
        return subscripts[tensorIDMap_r[tensorID]];
    }
    Range getIterRange(const Iterator &iter) const {
        return rangeOp->getRange(iter);
    }

    /**
     * @brief Check strides of each iterators and there position in tensors.
     * Since many-to-many iterators matching exist, we take this procudure as a
     * seperate function to deal with different iterator mapping solution.
     *
     * @param patternIT
     * @param mappingIter_r
     * @return vector<Mismatch>
     */
    vector<Mismatch>
    matchPatternITCheckStrides(const Pattern &patternIT,
                               PtrMap<Iterator, Iterator> mappingIter_r);
    RangeOp getRangeOp() const;
};

struct StrideConstraint {
    int tensorID;
    Var v0, v1;
    enum class Constraint { SAME, PROPOTIONAL } type;
};

class Pattern : public IteratorTable {
    vector<StrideConstraint> strideConstraints;

  public:
    virtual Expr
    buildExpr(const Expr &expr, const vector<Tensor> &tensors,
              [[maybe_unused]] const PatternIterRangeMap &varRanges,
              string outputName,
              [[maybe_unused]] const IteratorTable &exprIT) const = 0;
    /**
     * @brief Check whether all indexes only are a iterator
     *
     * @param tensorID
     */
    bool isAllUniqueAccess(int tensorID) const;
    const auto &getStrideConstraints() const { return strideConstraints; };
    int calcPadding(const Tensor &tensor, int dim, Range rangeH, Range rangeR,
                    int offset) const;
};

class MatmulPattern : public Pattern {
  public:
    static const Pattern &getMatmulPattern();
    static pair<Expr, pair<Tensor, Tensor>> getExpr(bool transA, bool transB,
                                                    int b, int m, int n, int k);

    Expr buildExpr(const Expr &expr, const vector<Tensor> &tensors,
                   [[maybe_unused]] const PatternIterRangeMap &varRanges,
                   string outputName,
                   [[maybe_unused]] const IteratorTable &exprIT) const override;
};

class ConvPattern : public Pattern {
  private:
    static const Var n, c, h, w, f, r, s;

  public:
    static const Pattern &getPattern();
    static Expr getExpr(Tensor A, Tensor K, int n, int c, int h, int w, int f,
                        int r, int s);

    Expr buildExpr(const Expr &expr, const vector<Tensor> &tensors,
                   [[maybe_unused]] const PatternIterRangeMap &varRanges,
                   string outputName,
                   [[maybe_unused]] const IteratorTable &exprIT) const override;
};

class ConvTransPattern : public Pattern {
  private:
    static const Var n, c, h, w, f, r, s;

  public:
    static const Pattern &getPattern() = delete;
    static Expr getExpr(Tensor A, Tensor K, int N, int C, int H, int W, int F,
                        int R, int S);

    Expr
    buildExpr(const Expr &expr, const vector<Tensor> &tensors,
              [[maybe_unused]] const PatternIterRangeMap &varRanges,
              string outputName,
              [[maybe_unused]] const IteratorTable &exprIT) const override {
        nnet_unimplemented_halt();
        return nullptr;
    };
};

class Sg2bmmPattern : public Pattern {
  private:
    static const Var b, m, w, k;

  public:
    static const Pattern &getPattern();
    static pair<Expr, pair<Tensor, Tensor>> getExpr(int Batch, int M, int K,
                                                    int W, int D);

    Expr buildExpr(const Expr &expr, const vector<Tensor> &tensors,
                   [[maybe_unused]] const PatternIterRangeMap &varRanges,
                   string outputName,
                   [[maybe_unused]] const IteratorTable &exprIT) const override;
};

class LongformerGBMMPattern : public Pattern {
  private:
    static const Var b, m, w, n;

  public:
    static const Pattern &getPattern();
    static pair<Expr, pair<Tensor, Tensor>> getExpr(int Batch, int M, int W,
                                                    int K, int dilation);

    Expr buildExpr(const Expr &expr, const vector<Tensor> &tensors,
                   [[maybe_unused]] const PatternIterRangeMap &varRanges,
                   string outputName,
                   [[maybe_unused]] const IteratorTable &exprIT) const override;
};

const Pattern &getPattern(RoutineType targetOp);
string getPatternName(RoutineType targetOp);

} // namespace nnet
