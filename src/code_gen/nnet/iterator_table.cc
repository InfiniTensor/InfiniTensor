#include "code_gen/nnet/iterator_table.h"
#include "code_gen/nnet/Visitor/MatchTableVisitor.h"
#include "code_gen/nnet/Visitor/SimplifyExprVisitor.h"
#include "code_gen/nnet/permutation.h"
#include <iostream>

namespace nnet {

bool IteratorTable::analyzeExpr(const RangeOp &_rangeOp) {
    rangeOp = _rangeOp;
    MatchTableVisitor mtVisitor;
    if (!mtVisitor(rangeOp))
        return false;
    tie(appearance, tensors, strideTable, subscripts) = mtVisitor.getResult();
    // dbg(appearance, tensors, strideTable);
    return true;
}

// mapTensors
void IteratorTable::buildTable(const vector<int> &_tensorMap) {
    tensorMap = _tensorMap;
    tensorMap_r.clear();
    tensorMap_r.resize(getNumInputs());
    tensorIDMap_r.clear();
    tensorIDMap_r.resize(getNumInputs());
    posTable.clear();
    posTable.resize(getNumRows());
    strideInTensor.clear();

    // build reversed index (from tensorID to tensor/original tensor index)
    for (size_t i = 0; i < tensorMap.size(); ++i) {
        tensorMap_r[tensorMap[i]] = tensors[i];
        tensorIDMap_r[tensorMap[i]] = i;
    }
    strideInDim.clear();
    strideInDim.resize(getNumInputs());
    for (int i = 0; i < getNumInputs(); ++i)
        strideInDim[i].resize(getTensor(i)->getDims());

    // auxiliary array for calculate in-dim stride
    vector<vector<int>> ldaInTensors(getNumInputs());
    for (int i = 0; i < getNumInputs(); ++i) {
        ldaInTensors[i].resize(getTensor(i)->getDims());
        ldaInTensors[i].back() = 1;
        for (int j = getTensor(i)->getDims() - 2; j >= 0; --j)
            ldaInTensors[i][j] =
                ldaInTensors[i][j + 1] * getTensor(i)->getShape(j + 1);
    }

    map<TensorNode *, int> inputTensor2id;
    for (int i = 0; i < getNumInputs(); ++i)
        inputTensor2id[tensors[i].get()] = tensorMap[i];

    iterInTensorDim.clear();
    iterInTensorDim.resize(getNumInputs());
    for (int i = 0; i < getNumInputs(); ++i) {
        iterInTensorDim[tensorMap[i]].resize(tensors[i]->getDims());
    }

    for (const auto &[var, tds] : strideTable) {
        int bitmap = 0;
        strideInTensor.emplace(var, getNumInputs());
        for (const auto &[tensorNode, dim, stride] : tds) {
            int tensorID = inputTensor2id[tensorNode];
            int bit = 1 << tensorID;
            if (std::find_if(iterInTensorDim[tensorID][dim].begin(),
                             iterInTensorDim[tensorID][dim].end(),
                             [var = var](const Var &v) {
                                 return v->equal(var);
                             }) == iterInTensorDim[tensorID][dim].end())
                iterInTensorDim[tensorID][dim].emplace_back(var);
            bitmap |= bit;
            if (strideInTensor[var][tensorID] < 0 || stride < 0)
                strideInTensor[var][tensorID] = -1;
            else
                strideInTensor[var][tensorID] += stride;
            // Update strideInDim
            assert(stride % ldaInTensors[tensorID][dim] == 0);
            strideInDim[tensorID][dim][var] =
                stride / ldaInTensors[tensorID][dim];
        }
        if (rangeOp->hasLoopVar(var))
            bitmap |= 1 << getNumInputs();
        posTable[bitmap].emplace_back(var);
    }
}

void IteratorTable::buildTableWithDefaultMap() {
    vector<int> tensorMap;
    for (int i = 0; i < getNumInputs(); ++i)
        tensorMap.emplace_back(i);
    buildTable(tensorMap);
}
int IteratorTable::getStridesInTensor(Iterator iter, int tensorID) const {
    return strideInTensor.at(iter).at(tensorID);
}

vector<int> IteratorTable::getIterDimInTensor(int tensorID,
                                              const Iterator &iter) const {
    vector<int> ret;
    for (size_t i = 0; i < iterInTensorDim[tensorID].size(); ++i) {
        for (const auto &it : iterInTensorDim[tensorID][i])
            if (iter->equal(it))
                ret.emplace_back(i);
    }
    return ret;
}

vector<Mismatch> IteratorTable::matchPatternIT(const Pattern &patternIT) {
    vector<Mismatch> ret;
    iterMap.clear();
    vector<vector<Iterator>> multiExprVar, multiPatternVar;
    // match iterators in single iterator rows
    for (int row = 0; row < getNumRows(); ++row) {
        int nExprVars = posTable[row].size(),
            nPatternVars = patternIT.posTable[row].size();
        if (nExprVars < nPatternVars) {
            ret.emplace_back(MismatchType::LessVar, row);
            continue;
        }
        if (nExprVars > nPatternVars) {
            ret.emplace_back(MismatchType::MoreVar, row);
            continue;
        }
        if (posTable[row].empty())
            continue;
        // prepare for many-to-many iterator mapping
        if (posTable[row].size() > 1) {
            multiExprVar.emplace_back(posTable[row]);
            multiPatternVar.emplace_back(patternIT.posTable[row]);
        }
        assert(!iterMap.count(posTable[row][0])); // check NO duplicate mapping
        if (posTable[row].size() == 1)
            iterMap[posTable[row][0]] = patternIT.posTable[row][0];
    }
    if (!ret.empty())
        return ret;
    PermutationGenerator permutationGenerator{multiPatternVar, multiExprVar};
    bool checked = false;
    // Permute iterator mappings to find a matched case
    do {
        auto mappingIter_r = permutationGenerator.get();
        for (const auto &[exprIter, patternIter] : iterMap)
            mappingIter_r[patternIter] = exprIter;
        auto mismatches = matchPatternITCheckStrides(patternIT, mappingIter_r);
        // if (mappingIter_r.count("_Conv_c"))
        //     if (mappingIter_r["_Conv_n"] == "n" &&
        //         mappingIter_r["_Conv_c"] == "c" &&
        //         mappingIter_r["_Conv_h"] == "i22" &&
        //         mappingIter_r["_Conv_r"] == "i4" &&
        //         mappingIter_r["_Conv_w"] == "i17" &&
        //         mappingIter_r["_Conv_s"] == "i14") {
        //         dbg(ret.size());
        //         if (mismatches.size() > 0)
        //             dbg(mismatches.size(), mismatches[0].type);
        //     }
        if (mismatches.size() == 0) { // matched
            ret = mismatches;
            // Complete iterator mapping
            for (const auto &[patternIter, exprIter] : mappingIter_r) {
                if (iterMap.count(exprIter))
                    assert(iterMap[exprIter]->equal(patternIter));
                iterMap[exprIter] = patternIter;
            }
            break;
        } else if (!checked) {
            ret = mismatches;
            checked = true;
        } else if ((static_cast<int>(ret[0].type) <
                    static_cast<int>(mismatches[0].type)) ||
                   ((static_cast<int>(ret[0].type) ==
                     static_cast<int>(mismatches[0].type)) &&
                    (mismatches.size() < ret.size()))) {
            ret = mismatches;
        }
    } while (permutationGenerator.next());
    // Build reverse iterator mapping
    if (ret.empty()) {
        iterToRange_r.clear();
        for (const auto &[exprIter, patternIter] : iterMap)
            iterToRange_r[patternIter] = rangeOp->getVarRange(exprIter);
    }
    return ret;
}

vector<Mismatch> IteratorTable::matchPatternITCheckStrides(
    const Pattern &patternIT, PtrMap<Iterator, Iterator> mappingIter_r) {
    vector<Mismatch> ret;
    // Check strides against each stride constraint
    for (const auto &constraint : patternIT.getStrideConstraints()) {
        // TODO: supprot PROPOTIONAL constraint
        auto stride0 = strideInTensor.at(
            mappingIter_r[constraint.v0])[constraint.tensorID];
        auto stride1 = strideInTensor.at(
            mappingIter_r[constraint.v1])[constraint.tensorID];
        if (stride0 != stride1) {
            ret.emplace_back(Mismatch(MismatchType::StrideMismatch, -1));
        }
    }
    if (!ret.empty())
        return ret;
    // check the appearance of iterators inside tensors.
    // If mismatch, this can be repaired by guided DLT.
    for (int tensorID = 0; tensorID < getNumInputs(); ++tensorID) {
        int exprTensorDim = tensorMap_r[tensorID]->getDims();
        int patternTensorDim = patternIT.tensorMap_r[tensorID]->getDims();
        if (exprTensorDim != patternTensorDim) {
            ret.emplace_back(MismatchType::DLMismatch, tensorID);
            continue;
        }
        [&] {
            for (int dim = 0; dim < exprTensorDim; ++dim) {
                // If #iters is differnt, than DLT is required
                if (strideInDim[tensorID][dim].size() !=
                    patternIT.strideInDim[tensorID][dim].size()) {
                    ret.emplace_back(MismatchType::DLMismatch, tensorID);
                    return;
                }
                for (const auto &[patternIter, patternStride] :
                     patternIT.strideInDim[tensorID][dim]) {
                    auto exprIter = mappingIter_r[patternIter];
                    // If iters are differnt
                    if (!strideInDim[tensorID][dim].count(exprIter)) {
                        ret.emplace_back(MismatchType::DLMismatch, tensorID);
                        return;
                    }
                    auto exprStride = strideInDim[tensorID][dim].at(exprIter);
                    // TODO: for stride and dilation
                    if (exprStride != patternStride) {
                        ret.emplace_back(MismatchType::DLMismatch, tensorID);
                        return;
                    }
                }
            }
        }();
    }
    if (!ret.empty())
        return ret;
    // check output data layout
    // Output dim mismatch is not implemented.
    if (patternIT.rangeOp->getNumOutputDims() != rangeOp->getNumOutputDims()) {
        ret.emplace_back(Mismatch{MismatchType::OutputDimismatch, 0});
        return ret;
    }
    for (size_t i = 0; i < rangeOp->getLoopVarRanges().size(); ++i) {
        if (!mappingIter_r[patternIT.rangeOp->getLoopVar(i)]->equal(
                rangeOp->getLoopVar(i))) {
            ret.emplace_back(MismatchType::OutputDLMismatch, getNumInputs(),
                             mappingIter_r);
            break;
        }
    }
    return ret;
}

pair<PatternTensorMap, PatternIterRangeMap>
IteratorTable::getReverseMap() const {
    return {tensorMap_r, iterToRange_r};
}

bool Pattern::isAllUniqueAccess(int tensorID) const {
    for (const auto &iterInDim : iterInTensorDim[tensorID]) {
        if (iterInDim.size() != 1)
            return false;
    }
    return true;
}

Expr MatmulPattern::buildExpr(
    const Expr &expr, const vector<Tensor> &tensors,
    [[maybe_unused]] const PatternIterRangeMap &varRanges, string outputName,
    [[maybe_unused]] const IteratorTable &exprIT) const {
    // TODO support b
    assert(tensors.size() == 2);
    int b = 1;
    int m = tensors[0]->getShape(0), n = tensors[1]->getShape(0);
    int k = tensors[0]->getShape(1);
    // TODO: check strides
    // TODO: DLT for output?
    // FIXME: check the trans
    auto matmul = make_ref<MatmulNode>(expr, tensors[0], tensors[1], b, m, n, k,
                                       false, true);
    auto output = make_ref<TensorNode>(outputName, vector<int>{m, n},
                                       vector<int>{0, 0}, matmul);
    return output;
}

const Pattern &MatmulPattern::getMatmulPattern() {
    static class MatmulPattern exprIT;
    static bool inited = false;
    if (!inited) {
        inited = true;
        int M = 224, N = 8, K = 16;
        auto m = make_ref<VarNode>("_Matmul_m");
        auto n = make_ref<VarNode>("_Matmul_n");
        auto k = make_ref<VarNode>("_Matmul_k");
        auto A = make_ref<TensorNode>("_Matmul_A", vector<int>({M, K}));
        auto B = make_ref<TensorNode>("_Matmul_B", vector<int>({N, K}));
        auto subA = makeSubscript(A, {m, k});
        auto subB = makeSubscript(B, {n, k});
        auto range = makeRangeOperator({{m, {0, M}}, {n, {0, N}}},
                                       {{k, {0, K}}}, subA * subB);
        auto success = exprIT.analyzeExpr(range);
        assert(success);
        exprIT.buildTable({0, 1});
    }
    return exprIT;
}

const Pattern &ConvPattern::getPattern() {
    static class ConvPattern exprIT;
    static bool inited = false;
    if (!inited) {
        inited = true;
        // The shape is meaningless but cannot be zero IT building
        int N = 8, C = 16, H = 224, W = 224, F = 16, R = 3, S = 3;
        // auto n = make_ref<VarNode>("_Matmul_n");
        auto A = make_ref<TensorNode>("_Conv_A", vector<int>({N, C, H, W}));
        auto B = make_ref<TensorNode>("_Conv_K", vector<int>({F, C, R, S}));
        auto subA = makeSubscript(A, {n, c, h + r, w + s});
        auto subB = makeSubscript(B, {f, c, r, s});
        auto range = makeRangeOperator(
            {{n, {0, 0}}, {f, {0, 0}}, {h, {0, 0}}, {w, {0, 0}}},
            {{c, {0, 0}}, {r, {0, 0}}, {s, {0, 0}}}, subA * subB);
        auto success = exprIT.analyzeExpr(range);
        assert(success);
        exprIT.buildTable({0, 1});
    }
    return exprIT;
}

Expr ConvPattern::buildExpr(
    const Expr &expr, const vector<Tensor> &tensors,
    const PatternIterRangeMap &varRanges, string outputName,
    [[maybe_unused]] const IteratorTable &exprIT) const {
    // calculate paddings
    const auto &rangeH = varRanges.at(h).second;
    const auto &rangeR = varRanges.at(r).second;
    const auto &rangeW = varRanges.at(w).second;
    const auto &rangeS = varRanges.at(s).second;
    auto offsetH =
        SimplifyExprVisitor().getConstant(exprIT.getSubscript(0)->getIndex(2));
    auto offsetW =
        SimplifyExprVisitor().getConstant(exprIT.getSubscript(0)->getIndex(3));
    int ph = calcPadding(tensors[0], 2, rangeH, rangeR, offsetH);
    int pw = calcPadding(tensors[0], 3, rangeW, rangeS, offsetW);

    // TODO strided, dilated
    auto conv = make_ref<ConvNode>(expr, tensors[0], tensors[1], ph, pw);
    auto shape = conv->getShape();
    auto rangeOpShape = as<RangeOpNode>(expr)->getOutputShape();
    assert(shape.size() == rangeOpShape.size());
    dbg(shape, rangeOpShape);
    for (size_t i = 0; i < shape.size(); ++i) {
        if (shape[i] != rangeOpShape[i]) {
            dbg("Warning: unmatched Conv output", shape, rangeOpShape);
            return nullptr;
        }
    }
    auto output =
        make_ref<TensorNode>(outputName, shape, vector<int>{0, 0, 0, 0}, conv);
    return output;
}

RangeOp IteratorTable::getRangeOp() const { return rangeOp; }

#define DEF_CONV_VAR(a)                                                        \
    const Var ConvPattern::a = make_ref<VarNode>("_Conv_" #a)
DEF_CONV_VAR(n);
DEF_CONV_VAR(c);
DEF_CONV_VAR(h);
DEF_CONV_VAR(w);
DEF_CONV_VAR(f);
DEF_CONV_VAR(r);
DEF_CONV_VAR(s);
#undef DEF_CONV_VAR

int Pattern::calcPadding(const Tensor &tensor, int dim, Range rangeH,
                         Range rangeR, int offset) const {
    int l = rangeH.first + rangeR.first + offset;
    int r = rangeH.second + rangeR.second - 1 + offset;
    int ret = max(0, max(0 - l, r - tensor->getShape(dim)));
    // dbg(l, r, rangeH, rangeR, offset, ret, tensor->getPadding(dim));
    // check OutOfBound
    assert(ret <= tensor->getPadding(dim));
    return ret;
}

#define DEF_SG2BMM_VAR(a)                                                      \
    const Var Sg2bmmPattern::a = make_ref<VarNode>("_Sg2bmm_" #a)
DEF_SG2BMM_VAR(b);
DEF_SG2BMM_VAR(m);
DEF_SG2BMM_VAR(w);
DEF_SG2BMM_VAR(k);
#undef DEF_SG2BMM_VAR

const Pattern &Sg2bmmPattern::getPattern() {
    static class Sg2bmmPattern exprIT;
    static bool inited = false;
    if (!inited) {
        inited = true;
        // The shape is meaningless but cannot be zero IT building
        int Batch = 8, M = 32, K = 224, W = 2;
        // auto n = make_ref<VarNode>("_Matmul_n");
        auto A = make_ref<TensorNode>("_Sg2bmm_A", vector<int>{Batch, M, K});
        auto B = make_ref<TensorNode>("_Sg2bmm_B", vector<int>{Batch, M, K});
        auto subA = makeSubscript(A, {b, m, k});
        auto subB = makeSubscript(B, {b, m + w, k});
        auto range =
            makeRangeOperator({{b, {0, Batch}}, {m, {0, M}}, {w, {-W, W + 1}}},
                              {{k, {0, K}}}, subA * subB);
        auto success = exprIT.analyzeExpr(range);
        assert(success);
        exprIT.buildTableWithDefaultMap();
    }
    return exprIT;
}

Expr Sg2bmmPattern::buildExpr(
    const Expr &expr, const vector<Tensor> &tensors,
    [[maybe_unused]] const PatternIterRangeMap &varRanges, string outputName,
    [[maybe_unused]] const IteratorTable &exprIT) const {
    // calculate paddings
    assert(tensors.size() == 2);
    assert(tensors[0]->getDims() == 3 && tensors[1]->getDims() == 3);
    int Batch = tensors[0]->getShape(0);
    int M = tensors[0]->getShape(1);
    int K = tensors[0]->getShape(2);
    int W = getLength(varRanges.at(w).second) / 2;

    auto op = make_ref<G2bmmNode>(expr, tensors[0], tensors[1], Batch, M, W, K);
    auto shape = op->getShape();
    auto rangeOpShape = as<RangeOpNode>(expr)->getOutputShape();
    assert(shape.size() == rangeOpShape.size());
    for (size_t i = 0; i < shape.size(); ++i) {
        assert(shape[i] == rangeOpShape[i]);
    }
    auto output =
        make_ref<TensorNode>(outputName, shape, vector<int>{0, 0, 0}, op);
    return output;
}

#define DEF_LongformerGBMM_VAR(a)                                              \
    const Var LongformerGBMMPattern::a = make_ref<VarNode>("_lo_" #a)
DEF_LongformerGBMM_VAR(b);
DEF_LongformerGBMM_VAR(m);
DEF_LongformerGBMM_VAR(w);
DEF_LongformerGBMM_VAR(n);
#undef DEF_LongformerGBMM_VAR

const Pattern &LongformerGBMMPattern::getPattern() {
    static class LongformerGBMMPattern exprIT;
    static bool inited = false;
    if (!inited) {
        inited = true;
        // The shape is meaningless but cannot be zero IT building
        int Batch = 8, M = 32, N = 224, W = 2;
        auto A =
            make_ref<TensorNode>("_lo_A", vector<int>{Batch, M, 2 * W + 1});
        auto B = make_ref<TensorNode>("_lo_B", vector<int>{Batch, M, N});
        auto subA = makeSubscript(A, {b, m, w});
        auto subB = makeSubscript(B, {b, m + w, n});
        auto range =
            makeRangeOperator({{b, {0, Batch}}, {m, {0, M}}, {n, {0, M}}},
                              {{w, {-W, W + 1}}}, subA * subB);
        auto success = exprIT.analyzeExpr(range);
        assert(success);
        exprIT.buildTableWithDefaultMap();
    }
    return exprIT;
}

Expr LongformerGBMMPattern::buildExpr(
    const Expr &expr, const vector<Tensor> &tensors,
    [[maybe_unused]] const PatternIterRangeMap &varRanges, string outputName,
    [[maybe_unused]] const IteratorTable &exprIT) const {
    // calculate paddings
    assert(tensors.size() == 2);
    assert(tensors[0]->getDims() == 3 && tensors[1]->getDims() == 3);
    int Batch = tensors[0]->getShape(0);
    int M = tensors[0]->getShape(1);
    assert(tensors[0]->getShape(2) % 2 == 1);
    int W = tensors[0]->getShape(2) / 2;
    int N = tensors[1]->getShape(2);

    auto op = make_ref<GbmmNode>(expr, tensors[0], tensors[1], Batch, M, W, N);
    auto shape = op->getShape();
    auto rangeOpShape = as<RangeOpNode>(expr)->getOutputShape();
    assert(shape.size() == rangeOpShape.size());
    for (size_t i = 0; i < shape.size(); ++i) {
        assert(shape[i] == rangeOpShape[i]);
    }
    auto output = make_ref<TensorNode>(outputName, shape,
                                       vector<int>(shape.size(), 0), op);
    return output;
}

const Pattern &getPattern(RoutineType targetOp) {
    switch (targetOp) {
    case RoutineType::MatmulNodeType:
        return MatmulPattern::getMatmulPattern();
    case RoutineType::ConvNodeType:
        return ConvPattern::getPattern();
    case RoutineType::G2bmmNodeType:
        return Sg2bmmPattern::getPattern();
    case RoutineType::GbmmNodeType:
        return LongformerGBMMPattern::getPattern();
    default:
        nnet_unimplemented_halt();
    }
}

string getPatternName(RoutineType targetOp) {
    switch (targetOp) {
    case RoutineType::MatmulNodeType:
        return "Matmul";
    case RoutineType::ConvNodeType:
        return "Conv";
    case RoutineType::G2bmmNodeType:
        return "G2bmm";
    case RoutineType::GbmmNodeType:
        return "Gbmm";
    default:
        nnet_unimplemented_halt();
    }
    return {};
}

#define DEFINE_VAR(name) auto name = make_ref<VarNode>(#name);
Expr ConvPattern::getExpr(Tensor A, Tensor K, int N, int C, int H, int W, int F,
                          int R, int S) {
    DEFINE_VAR(n);
    DEFINE_VAR(c);
    DEFINE_VAR(h);
    DEFINE_VAR(w);
    DEFINE_VAR(f);
    DEFINE_VAR(r);
    DEFINE_VAR(s);
    auto subA = makeSubscript(A, {n, c, h + r - R / 2, w + s - S / 2});
    auto subB = makeSubscript(K, {f, c, r, s});
    auto range =
        makeRangeOperator({{n, {0, N}}, {f, {0, F}}, {h, {0, H}}, {w, {0, W}}},
                          {{c, {0, C}}, {r, {0, R}}, {s, {0, S}}}, subA * subB);
    return range;
}

// Warn: F is the number of input channels, which is inversed compared with
// normal Conv.
// Input / output layouts: NHWF -> NHWC, Kenrel: RSFC
Expr ConvTransPattern::getExpr(Tensor A, Tensor K, int N, int C, int H, int W,
                               int F, int R, int S) {
    const int padding = 1 * (R - 1) - 1;
    assert(A->getPadding(2) == padding);
    assert(R == 4);
    assert(S == 4);
    const int OH = 2 * H, OW = 2 * W;
    DEFINE_VAR(n);
    DEFINE_VAR(c);
    DEFINE_VAR(f);
    DEFINE_VAR(r);
    DEFINE_VAR(s);
    DEFINE_VAR(x1);
    DEFINE_VAR(x2);
    DEFINE_VAR(y1);
    DEFINE_VAR(y2);
    DEFINE_VAR(i2);
    DEFINE_VAR(i4);
    DEFINE_VAR(h);
    DEFINE_VAR(w);
    // dilation * (kernel_size - 1) - padding
    // auto A = make_ref<TensorNode>("A", vector<int>({N, H, W, F}),
    //                               vector<int>{0, padding, padding, 0});
    // auto K = make_ref<TensorNode>("K", vector<int>({R, S, F, C}));

    auto subA = makeSubscript(A, {n, x1 + r - 1, y1 + s - 1, f});
    auto subK =
        makeSubscript(K, {(R - 2) - 2 * r + x2, (S - 2) - 2 * s + y2, f, c});
    // x1=(h+1)//2, x2=(h+1)%2, y1=(w+1)//2

    auto range1 = makeRangeOperator(
        {
            {n, {0, N}},
            {c, {0, C}},
            {x1, {0, OH / 2 + 1}},
            {x2, {0, 2}},
            {y1, {0, OW / 2 + 1}},
            {y2, {0, 2}},
        },
        {{f, {0, F}}, {r, {0, R / 2}}, {s, {0, S / 2}}}, subA * subK);
    auto sub0 = makeSubscript(
        range1, {n, c, (h + 1) / 2, (h + 1) % 2, (w + 1) / 2, (w + 1) % 2});
    auto range0 = makeRangeOperator(
        {{n, {0, N}}, {h, {0, OH}}, {w, {0, OW}}, {c, {0, C}}}, {}, sub0);
    return range0;
}

pair<Expr, pair<Tensor, Tensor>> Sg2bmmPattern::getExpr(int Batch, int M, int K,
                                                        int W, int D) {
    DEFINE_VAR(b);
    DEFINE_VAR(m);
    DEFINE_VAR(w);
    DEFINE_VAR(k);
    auto A = make_ref<TensorNode>("A", vector<int>({Batch, M, K}),
                                  vector<int>{0, 0, 0});
    auto B = make_ref<TensorNode>("B", vector<int>({Batch, M, K}),
                                  vector<int>{0, D * W, 0});

    auto subA = makeSubscript(A, {b, m, k});
    auto subB = makeSubscript(B, {b, m + D * (w - W), k});
    auto range =
        makeRangeOperator({{b, {0, Batch}}, {m, {0, M}}, {w, {0, 2 * W + 1}}},
                          {{k, {0, K}}}, subA * subB);
    return {range, {A, B}};
}

pair<Expr, pair<Tensor, Tensor>>
LongformerGBMMPattern::getExpr(int Batch, int M, int W, int K, int dilation) {
    DEFINE_VAR(b);
    DEFINE_VAR(m);
    DEFINE_VAR(w);
    DEFINE_VAR(k);
    auto A = make_ref<TensorNode>("A", vector<int>({Batch, M, 2 * W + 1}),
                                  vector<int>{0, 0, 0});
    auto B = make_ref<TensorNode>("B", vector<int>({Batch, M, K}),
                                  vector<int>{0, dilation * W, 0});
    auto subA = makeSubscript(A, {b, m, w});
    auto subB = makeSubscript(B, {b, m + dilation * w - dilation * W, n});
    auto range = makeRangeOperator({{b, {0, Batch}}, {m, {0, M}}, {n, {0, K}}},
                                   {{w, {0, 2 * W + 1}}}, subA * subB);
    return {range, {A, B}};
}

pair<Expr, pair<Tensor, Tensor>> MatmulPattern::getExpr(bool transA,
                                                        bool transB, int Batch,
                                                        int M, int N, int K) {
    DEFINE_VAR(b);
    DEFINE_VAR(m);
    DEFINE_VAR(n);
    DEFINE_VAR(k);
    auto A = make_ref<TensorNode>("A", vector<int>({Batch, M, K}),
                                  vector<int>{0, 0, 0});
    auto B = make_ref<TensorNode>("B", vector<int>({Batch, K, N}),
                                  vector<int>{0, 0, 0});
    auto subA = makeSubscript(A, {b, m, k});
    auto subB = makeSubscript(B, {b, k, n});
    auto range = makeRangeOperator({{b, {0, Batch}}, {m, {0, M}}, {n, {0, N}}},
                                   {{k, {0, K}}}, subA * subB);
    return {range, {A, B}};
}

#undef DEFINE_VAR

} // namespace nnet
