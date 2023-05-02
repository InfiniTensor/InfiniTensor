#include "pet/pmutator.h"
#include "core/graph.h"
#include "cuda/cuda_runtime.h"
#include "ffi/ffi_callback.h"
#include "operators/G2BMM.h"
#include "operators/GBMM.h"
#include "operators/any.h"
#include "operators/conv.h"
#include "operators/conv2dreduce.h"
#include "operators/matmul.h"
#include "operators/membound.h"
#include "operators/reduce_mean.h"
#include "operators/reshape.h"
#include "operators/softmax.h"
#include "operators/transpose.h"
#include "operators/unary.h"

namespace infini {

PMutator::PMutator(Mode mode, Runtime runtime)
    : Mutator(10, runtime), mode{mode} {
    IT_ASSERT(mode != Mode::RuleBased, "Specify rules for the RuleBased mode.");
}

PMutator::~PMutator() {}

vector<Graph> PMutator::run(const Graph &in_graph) {
    vector<Graph> out_graphs{in_graph};
    OpVec computeOps = in_graph->getComputeOps();
    // assert(computeOps.size() == 1);
    if (computeOps.size() == 1)
        runSingleOp(in_graph, out_graphs);
    // FIXME: runMultipleOps results in segfault
    // else
    //     runMultipleOps(in_graph, out_graphs);
    return out_graphs;
}

bool PMutator::isMultiBranchMergable(const Graph &in_graph) {
    // TODO
    // dbg("Skip mergable Multi-Branch", in_graph);
    return false;
}

void PMutator::runSingleOp(Graph in_graph, std::vector<Graph> &out_graphs) {
    OpVec computeOps = in_graph->getComputeOps();
    IT_ASSERT(computeOps.size() == 1);
    for (auto g : transformConv1x1(computeOps[0]))
        out_graphs.emplace_back(g);
    if (infini::Graph g = transformDialtedConv(computeOps[0]))
        out_graphs.emplace_back(g);
    if (auto g = transformConvW2N(computeOps[0]))
        out_graphs.emplace_back(g);
    if (auto g = transformConvH2N(computeOps[0]))
        out_graphs.emplace_back(g);
    if (auto g = transformConvH2W(computeOps[0]))
        out_graphs.emplace_back(g);
    if (auto g = transformConvW2H(computeOps[0]))
        out_graphs.emplace_back(g);
    if (auto g = transformConvN2H(computeOps[0]))
        out_graphs.emplace_back(g);
    if (auto g = transformConvN2W(computeOps[0]))
        out_graphs.emplace_back(g);
}

void PMutator::runMultipleOps(Graph in_graph, std::vector<Graph> &out_graphs) {
    IT_TODO_HALT();
}

double PMutator::memboundTime(ssize_t cnt) {
    return double(cnt) * 4 / bandwidth * 1000; // millisecond
}

double PMutator::memboundTime(const Shape &dims) {
    return memboundTime(dims.size());
}

Graph PMutator::transformDialtedConv(Operator _op) {
    auto op = as<ConvObj>(_op);
    if (!op)
        return {};
    const auto &[ph, pw, sh, sw, dh, dw] = op->getPadStrideDilation();
    if (dh <= 1 && dw <= 1)
        return {};
    const auto &A = op->getInputs()[0];
    const auto &W = op->getInputs()[1];
    const auto &O = op->getOutput();
    const auto &[n, c, h, w, f, r, s] = op->getNCHWFRS();
    const Shape inputDims = A->getDims();
    const Shape weightDims = W->getDims();
    const Shape outputDims = O->getDims();
    const DataType dtype = A->getDType();
    IT_ASSERT_TODO(dh == dw && ph == dh && pw == dw);
    IT_ASSERT_TODO(tie(sh, sw) == tuple(1, 1));
    IT_ASSERT_TODO(h % dh == 0 && w % dw == 0);
    auto g = make_ref<GraphObj>(runtime);
    auto newA = g->addTensor({dh * dw * n, c, h / dh, h / dw}, dtype);
    // HACH: without transpose
    g->addOpWithOutputs<ReshapeObj>(g->cloneTensor(A), newA, newA->getDims());
    Tensor newO =
        g->addOp<ConvObj>(newA, W, nullptr, 1, 1, sh, sw, 1, 1)->getOutput();
    // HACH: without transpose
    g->addOpWithOutputs<ReshapeObj>(newO, g->cloneTensor(O), O->getDims());
    return g;
}

vector<Graph> PMutator::transformConv1x1(Operator _op) {
    vector<Graph> ret;
    auto op = as<ConvObj>(_op);
    if (!op)
        return {};
    const auto &[ph, pw, sh, sw, dh, dw] = op->getPadStrideDilation();
    Shape shapeA = op->getInputs(0)->getDims();
    Shape shapeW = op->getInputs(1)->getDims();
    Shape shapeO = op->getOutput()->getDims();
    if (sh != 1 || sw != 1 || dh != 1 || dw != 1 || shapeW[2] != 1 ||
        shapeW[3] != 1)
        return {};
    if (shapeA[0] == 1) {
        {
            auto g = make_ref<GraphObj>(runtime);
            auto A =
                g->addOp<ReshapeObj>(g->cloneTensor(op->getInputs(0)), nullptr,
                                     vector{shapeA[1], shapeA[2] * shapeA[3]})
                    ->getOutput(); // [C, H*W]
            auto B =
                g->addOp<ReshapeObj>(g->cloneTensor(op->getInputs(1)), nullptr,
                                     vector{shapeW[0], shapeW[1]}) // [F, C]
                    ->getOutput();
            auto O = g->addOp<MatmulObj>(B, A, nullptr, false, false)
                         ->getOutput(); // [F, N*H*W]
            g->addOpWithOutputs<ReshapeObj>(O, g->cloneTensor(op->getOutput()),
                                            op->getOutput()->getDims());
            ret.emplace_back(g);
        }
        {
            auto g = make_ref<GraphObj>(runtime);
            auto A = g->addOp<ReshapeObj>(
                          g->cloneTensor(op->getInputs(0)), nullptr,
                          vector{shapeA[1], shapeA[2] * shapeA[3]}) // [C, HW]
                         ->getOutput();
            auto B = g->addOp<TransposeObj>(
                          g->cloneTensor(op->getInputs(1)), //[C,F,1,1]
                          nullptr, vector{1, 0, 2, 3})
                         ->getOutput();
            B = g->addOp<ReshapeObj>(B, nullptr,
                                     vector{shapeW[1], shapeW[0]}) // [C, F]
                    ->getOutput();
            auto O = g->addOp<MatmulObj>(B, A, nullptr, true, false)
                         ->getOutput(); // [F, N*H*W]
            g->addOpWithOutputs<ReshapeObj>(O, g->cloneTensor(op->getOutput()),
                                            op->getOutput()->getDims());
            ret.emplace_back(g);
        }
        // } else { // Tranpose + Matmul + Transpose
        //     auto A = g->addOp<TransposeObj>(g->cloneTensor(op->getInputs(0)),
        //                                     nullptr, vector{1, 0, 2, 3})
        //                  ->getOutput(); // [C,N,H,W]
        //     A = g->addOp<ReshapeObj>(A, nullptr,
        //                              vector{shapeA[1], shapeA[0] * shapeA[2]
        //                              *
        //                                                    shapeA[3]}) // [C,
        //                                                    N*H*W]
        //             ->getOutput();
        //     auto B = g->addOp<ReshapeObj>(g->cloneTensor(op->getInputs(1)),
        //     nullptr,
        //                                   vector{shapeW[0], shapeW[1]}) //
        //                                   [F, C]
        //                  ->getOutput();
        //     auto O =
        //         g->addOp<MatmulObj>(B, A, nullptr, 0, 0)->getOutput(); // [F,
        //         NHW]
        //     O = g->addOp<ReshapeObj>(
        //              O, nullptr, Shape{shapeO[1], shapeO[0], shapeO[2],
        //              shapeO[3]})
        //             ->getOutput(); // [F, NHW]
        //     O = g->addOpWithOutputs<TransposeObj>(
        //              O, g->cloneTensor(op->getOutput()), vector{1, 0, 2, 3})
        //             ->getOutput(); // [F, N*H*W]
    } else { // BGemm
        auto g = make_ref<GraphObj>(runtime);
        auto A =
            g->addOp<ReshapeObj>(g->cloneTensor(op->getInputs(0)), nullptr,
                                 vector{shapeA[0], shapeA[1],
                                        shapeA[2] * shapeA[3]}) // [N, C, H*W]
                ->getOutput();
        auto B =
            g->addOp<ReshapeObj>(g->cloneTensor(op->getInputs(1)), nullptr,
                                 vector{1, shapeW[0], shapeW[1]}) // [1, F, C]
                ->getOutput();
        auto O =
            g->addOp<MatmulObj>(B, A, nullptr, 0, 0)->getOutput(); // [F, N*H*W]
        g->addOpWithOutputs<ReshapeObj>(O, g->cloneTensor(op->getOutput()),
                                        op->getOutput()->getDims());
        ret.emplace_back(g);
    }
    return ret;
}

Graph PMutator::constructGraphByOperatorChain(vector<Operator> ops,
                                              Graph inputGraph) {
    // Construct new graph
    auto g = make_ref<GraphObj>(runtime);
    IT_ASSERT(inputGraph->getInputs().size() == 1);
    IT_ASSERT(inputGraph->getOutputs().size() == 1);
    auto input = g->cloneTensor(inputGraph->getInputs()[0]);
    auto graphOutput = g->cloneTensor(inputGraph->getOutputs()[0]);
    for (size_t i = 0; i < ops.size(); ++i) {
        if (i + 1 == ops.size() &&
            ops[i]->getOutput()->getDims() == graphOutput->getDims()) {
            input =
                g->cloneOperator(ops[i], {input}, {graphOutput})->getOutput();
        } else { // If it is not the last op or output shape dismatches
            input = g->cloneOpAndCreateOutputs(ops[i], {input})->getOutput();
        }
    }
    // Add a reshape to match original graph if necessary
    // HACK: If ops is an empty vector, add a reshape operator to construct a
    // valid return graph.
    if (g->getOutputs()[0]->getDims() != graphOutput->getDims() || ops.empty())
        g->addOpWithOutputs<ReshapeObj>(input, graphOutput);
    return g;
}

Graph PMutator::eliminateVertically(const Graph &inputGraph) {
    auto ops = inputGraph->getOperators();
    bool funcHasOptmization = false;

    IT_ASSERT(!ops.empty());
    for (auto &op : ops) {
        IT_ASSERT(op->isMemBoundOp());
        IT_ASSERT_TODO(op->getInputs().size() == 1);
        IT_ASSERT(op->getOutputs().size() == 1);
    }
    if (ops.size() == 1) {
        return make_ref<GraphObj>(runtime, ops);
    }

    // Set attributs for operators.
    // isComputation: is computaiton
    // isElementwise: do elementwise computations
    // lastRowSwapable: do last-channel-wise computations, which includes
    // elementwise as a special case.
    auto classifyOperator = [](Operator op) {
        auto type = op->getOpType();
        bool isComputation =
            type != OpType::Reshape && type != OpType::Transpose;
        bool isElementwise =
            !isComputation || (type == OpType::Relu || type == OpType::Tanh);
        bool lastRowSwapable = false;
        if (isComputation)
            lastRowSwapable = isElementwise || // Softmax along the last dim
                              (type == OpType::Softmax &&
                               as<SoftmaxObj>(op)->getAxis() ==
                                   int(op->getOutput()->getDims().size()) - 1);
        else {
            if (auto t = as<TransposeObj>(op)) {
                // Last dim remains unchanged
                lastRowSwapable =
                    (t->getPermute().back() == int(t->getPermute().size()) - 1);
            } else if (auto t = as<ReshapeObj>(op)) {
                // Last dim remains unchanged
                lastRowSwapable = (t->getInputs(0)->getDims().back() ==
                                   t->getOutput()->getDims().back());
            }
        }
        return tuple{isComputation, isElementwise, lastRowSwapable};
    };

    // Reorder operators: move computatation operators to the head
    for (int i = ops.size() - 2; i >= 0; --i) {
        for (int j = i; j < int(ops.size()) - 1; ++j) {
            bool swapable = false;
            auto [aIsC, aEw, aLRS] = classifyOperator(ops[j]);
            auto [bIsC, bEw, bLRS] = classifyOperator(ops[j + 1]);
            // check swapable conditions:
            // (!aIsC && bIsC): ordering of computation and non-computation
            // (aEw && aEw): elementwise
            // (aLRS && bLRS): last dim fixed
            if ((!aIsC && bIsC) && ((aEw && bEw) || (aLRS && bLRS)))
                swapable = true;
            if (swapable) {
                swap(ops[j], ops[j + 1]);
            }
        }
    }

    Graph g = constructGraphByOperatorChain(ops, inputGraph);
    // Eliminate operators
    bool haveElimination = false;
    do {
        funcHasOptmization = funcHasOptmization || haveElimination;
        haveElimination = false;
        ops = g->getOperators();
        // HACK: constructGraphByOperatorChain function will return a graph with
        // a reshape operator when all the operators are eliminated, because we
        // cannot express an empty graph by now.
        if (ops.size() == 1) {
            break;
        }
        vector<Operator> newOps;
        for (int i = 0; i < int(ops.size()); ++i) {
            // Eliminate identity operators
            if (auto op = as<TransposeObj>(ops[i])) {
                auto perm = op->getPermute();
                int j = 0;
                for (j = 0; j < int(perm.size()); ++j)
                    if (j != perm[j])
                        break;
                if (j == int(perm.size())) {
                    haveElimination = true;
                    continue;
                }
            } else if (auto op = as<ReshapeObj>(ops[i])) {
                if (op->getShape() == op->getInputs(0)->getDims()) {
                    haveElimination = true;
                    continue;
                }
            }

            // Operator-level fusion
            // Any+Relu -> Any(activation=1)
            if (i + 1 < int(ops.size())) {
                const string name = "reduceConvRxSToNCHW";
                if (auto op = as<AnyObj>(ops[i]);
                    op && op->getKernelName() == name) {
                    if (auto op2 = as<ReluObj>(ops[i + 1])) {
                        if (op->getOutput() == op2->getInputs(0)) {
                            auto newOp = make_ref<AnyObj>(*op);
                            newOp->setAttr(0, 1); // Set activation
                            newOps.push_back(newOp);
                            ++i;
                            haveElimination = true;
                            continue;
                        }
                    }
                }
            }

            // Eliminate reciprocal operators
            if (i + 1 == (int)ops.size() ||
                (ops[i]->getOpType() != ops[i + 1]->getOpType())) {
                newOps.push_back(ops[i]);
                continue;
            }
            if (ops[i]->getOpType() == OpType::Reshape) {
                newOps.push_back(make_ref<ReshapeObj>(
                    nullptr, ops[i]->getInputs(0), ops[i + 1]->getOutput()));
                ++i;
                haveElimination = true;
            } else if (ops[i]->getOpType() == OpType::Transpose) {
                auto permuteA = as<TransposeObj>(ops[i])->getPermute();
                auto permuteB = as<TransposeObj>(ops[i + 1])->getPermute();
                vector<int> permute;
                for (auto p : permuteB)
                    permute.push_back(permuteA[p]);
                newOps.push_back(
                    make_ref<TransposeObj>(nullptr, ops[i]->getInputs(0),
                                           ops[i + 1]->getOutput(), permute));
                ++i;
                haveElimination = true;
            } else {
                newOps.push_back(ops[i]);
            }
        }
        g = constructGraphByOperatorChain(newOps, inputGraph);
    } while (haveElimination);
    return g;
}

Graph PMutator::fuseVertically(const Graph &inputGraph) { return inputGraph; }

Tensor PMutator::splitTransposeMerge(Graph g, Tensor A, int dim, int chunkSize,
                                     Tensor output) {
    IT_ASSERT(A->getDims().size() == 3);
    Shape shapeOrignial = A->getDims();
    Shape shapeNew;
    // Construct new shape
    for (int i = 0; i < dim; ++i)
        shapeNew.emplace_back(shapeOrignial[i]);
    shapeNew.emplace_back(shapeOrignial[dim] / chunkSize);
    shapeNew.emplace_back(chunkSize);
    for (size_t i = dim + 1; i < shapeOrignial.size(); ++i)
        shapeNew.emplace_back(shapeOrignial[i]);
    auto A1 = g->addOp<ReshapeObj>(A, nullptr, shapeNew)->getOutput();
    auto A2 =
        g->addOp<TransposeObj>(A1, nullptr, vector{0, 2, 1, 3})->getOutput();
    Tensor A3;
    if (output)
        A3 = g->addOpWithOutputs<ReshapeObj>(A2, output, shapeOrignial)
                 ->getOutput();
    else
        A3 = g->addOp<ReshapeObj>(A2, nullptr, shapeOrignial)->getOutput();
    return A3;
};

Graph PMutator::transformConvW2N(Operator _op) {
    auto op = as<ConvObj>(_op);
    if (!op)
        return nullptr;
    const auto &A = op->getInputs()[0];
    const auto &W = op->getInputs()[1];
    const auto &O = op->getOutput();
    const auto &[n, c, h, w, f, r, s] = op->getNCHWFRS();
    const auto &[ph, pw, sh, sw, dh, dw] = op->getPadStrideDilation();
    auto dtype = A->getDType();
    if (dh != 1 || dw != 1 || sh != 1 || sw != 1)
        return nullptr;
    if (w % 2 != 0)
        return nullptr;

    auto g = make_ref<GraphObj>(runtime);
    auto newA = g->addTensor({n * 2, c, h, w / 2}, dtype);
    // HACK: use ReshapeOp instead of Reshape-Transpose-Reshape
    g->addOpWithOutputs<ReshapeObj>(g->cloneTensor(A), newA, newA->getDims());
    auto newO = g->addOp<ConvObj>(newA, g->cloneTensor(W), nullptr, ph, pw, sh,
                                  sw, dh, dw)
                    ->getOutput();
    // HACK: use ReshapeOp instead of Reshape-Transpose-Reshape
    g->addOpWithOutputs<ReshapeObj>(newO, g->cloneTensor(O), O->getDims());
    return g;
}

Graph PMutator::transformConvH2N(Operator _op) {
    auto op = as<ConvObj>(_op);
    if (!op)
        return nullptr;
    const auto &A = op->getInputs()[0];
    const auto &W = op->getInputs()[1];
    const auto &O = op->getOutput();
    const auto &[n, c, h, w, f, r, s] = op->getNCHWFRS();
    const auto &[ph, pw, sh, sw, dh, dw] = op->getPadStrideDilation();
    auto dtype = A->getDType();
    if (dh != 1 || dw != 1 || sh != 1 || sw != 1)
        return nullptr;
    if (h % 2 != 0)
        return nullptr;

    auto g = make_ref<GraphObj>(runtime);
    auto newA = g->addTensor({n * 2, c, h / 2, w}, dtype);
    // HACK: use ReshapeOp instead of Reshape-Transpose-Reshape
    g->addOpWithOutputs<ReshapeObj>(g->cloneTensor(A), newA, newA->getDims());
    auto newO = g->addOp<ConvObj>(newA, g->cloneTensor(W), nullptr, ph, pw, sh,
                                  sw, dh, dw)
                    ->getOutput();
    // HACK: use ReshapeOp instead of Reshape-Transpose-Reshape
    g->addOpWithOutputs<ReshapeObj>(newO, g->cloneTensor(O), O->getDims());
    return g;
}

Graph PMutator::transformConvH2W(Operator _op) {
    auto op = as<ConvObj>(_op);
    if (!op)
        return nullptr;
    const auto &A = op->getInputs()[0];
    const auto &W = op->getInputs()[1];
    const auto &O = op->getOutput();
    const auto &[n, c, h, w, f, r, s] = op->getNCHWFRS();
    const auto &[ph, pw, sh, sw, dh, dw] = op->getPadStrideDilation();
    auto dtype = A->getDType();
    if (dh != 1 || dw != 1 || sh != 1 || sw != 1)
        return nullptr;
    if (h % 2 != 0)
        return nullptr;

    auto g = make_ref<GraphObj>(runtime);
    auto newA = g->addTensor({n, c, h / 2, w * 2}, dtype);
    // HACK: use ReshapeOp instead of Reshape-Transpose-Reshape
    g->addOpWithOutputs<ReshapeObj>(g->cloneTensor(A), newA, newA->getDims());
    auto newO = g->addOp<ConvObj>(newA, g->cloneTensor(W), nullptr, ph, pw, sh,
                                  sw, dh, dw)
                    ->getOutput();
    // HACK: use ReshapeOp instead of Reshape-Transpose-Reshape
    g->addOpWithOutputs<ReshapeObj>(newO, g->cloneTensor(O), O->getDims());
    return g;
}

Graph PMutator::transformConvW2H(Operator _op) {
    auto op = as<ConvObj>(_op);
    if (!op)
        return nullptr;
    const auto &A = op->getInputs()[0];
    const auto &W = op->getInputs()[1];
    const auto &O = op->getOutput();
    const auto &[n, c, h, w, f, r, s] = op->getNCHWFRS();
    const auto &[ph, pw, sh, sw, dh, dw] = op->getPadStrideDilation();
    auto dtype = A->getDType();
    if (dh != 1 || dw != 1 || sh != 1 || sw != 1)
        return nullptr;
    if (w % 2 != 0)
        return nullptr;

    auto g = make_ref<GraphObj>(runtime);
    auto newA = g->addTensor({n, c, h * 2, w / 2}, dtype);
    // HACK: use ReshapeOp instead of Reshape-Transpose-Reshape
    g->addOpWithOutputs<ReshapeObj>(g->cloneTensor(A), newA, newA->getDims());
    auto newO = g->addOp<ConvObj>(newA, g->cloneTensor(W), nullptr, ph, pw, sh,
                                  sw, dh, dw)
                    ->getOutput();
    // HACK: use ReshapeOp instead of Reshape-Transpose-Reshape
    g->addOpWithOutputs<ReshapeObj>(newO, g->cloneTensor(O), O->getDims());
    return g;
}

Graph PMutator::transformConvN2H(Operator _op) {
    auto op = as<ConvObj>(_op);
    if (!op)
        return nullptr;
    const auto &A = op->getInputs()[0];
    const auto &W = op->getInputs()[1];
    const auto &O = op->getOutput();
    const auto &[n, c, h, w, f, r, s] = op->getNCHWFRS();
    const auto &[ph, pw, sh, sw, dh, dw] = op->getPadStrideDilation();
    auto dtype = A->getDType();
    if (dh != 1 || dw != 1 || sh != 1 || sw != 1)
        return nullptr;
    if (n % 2 != 0)
        return nullptr;

    auto g = make_ref<GraphObj>(runtime);
    auto newA = g->addTensor({n / 2, c, h * 2, w}, dtype);
    // HACK: use ReshapeOp instead of Reshape-Transpose-Reshape
    g->addOpWithOutputs<ReshapeObj>(g->cloneTensor(A), newA, newA->getDims());
    auto newO = g->addOp<ConvObj>(newA, g->cloneTensor(W), nullptr, ph, pw, sh,
                                  sw, dh, dw)
                    ->getOutput();
    // HACK: use ReshapeOp instead of Reshape-Transpose-Reshape
    g->addOpWithOutputs<ReshapeObj>(newO, g->cloneTensor(O), O->getDims());
    return g;
}

Graph PMutator::transformConvN2W(Operator _op) {
    auto op = as<ConvObj>(_op);
    if (!op)
        return nullptr;
    const auto &A = op->getInputs()[0];
    const auto &W = op->getInputs()[1];
    const auto &O = op->getOutput();
    const auto &[n, c, h, w, f, r, s] = op->getNCHWFRS();
    const auto &[ph, pw, sh, sw, dh, dw] = op->getPadStrideDilation();
    auto dtype = A->getDType();
    if (dh != 1 || dw != 1 || sh != 1 || sw != 1)
        return nullptr;
    if (n % 2 != 0)
        return nullptr;

    auto g = make_ref<GraphObj>(runtime);
    auto newA = g->addTensor({n / 2, c, h, w * 2}, dtype);
    // HACK: use ReshapeOp instead of Reshape-Transpose-Reshape
    g->addOpWithOutputs<ReshapeObj>(g->cloneTensor(A), newA, newA->getDims());
    auto newO = g->addOp<ConvObj>(newA, g->cloneTensor(W), nullptr, ph, pw, sh,
                                  sw, dh, dw)
                    ->getOutput();
    // HACK: use ReshapeOp instead of Reshape-Transpose-Reshape
    g->addOpWithOutputs<ReshapeObj>(newO, g->cloneTensor(O), O->getDims());
    return g;
}

} // namespace infini
