#include "nnet/nmutator.h"
#include "core/graph.h"
#include "cuda/cuda_runtime.h"
#include "ffi/ffi_callback.h"
#include "nnet/Visitor/FullPrinterVisitor.h"
#include "nnet/Visitor/GetTensorsVisitor.h"
#include "nnet/Visitor/MatchReshapeVisitor.h"
#include "nnet/Visitor/MergeMemboundMutator.h"
#include "nnet/derivator.h"
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

NMutator::NMutator(Mode mode, Runtime runtime)
    : Mutator(10, runtime), mode{mode} {
    IT_ASSERT(mode != Mode::RuleBased, "Specify rules for the RuleBased mode.");
}

NMutator::NMutator(Mode mode, const std::vector<int> &derivationRules,
                   Runtime runtime)
    : Mutator(10, runtime), mode{Mode::RuleBased}, derivationRules{
                                                       derivationRules} {
    IT_ASSERT(mode == Mode::RuleBased);
}

NMutator::~NMutator() {}

void NMutator::setToNaiveMembound() { mode = Mode::ToNaiveMembound; }

vector<Graph> NMutator::run(const Graph &in_graph) {
    vector<Graph> out_graphs{in_graph};
    // Test helper: naively transform one Op to Membound
    if (mode == Mode::ToNaiveMembound) {
        runSingleOpToNaiveMembound(in_graph, out_graphs);
        return out_graphs;
    }
    // Clear input names maps with tensor
    inputsNameNToTensorT.clear();
    OpVec computeOps = in_graph->getComputeOps();
    // assert(computeOps.size() == 1);
    if (computeOps.size() == 1)
        runSingleOp(in_graph, out_graphs);
    // FIXME: runMultipleOps results in segfault
    // else
    //     runMultipleOps(in_graph, out_graphs);
    return out_graphs;
}

bool NMutator::isMultiBranchMergable(const Graph &in_graph) {
    // TODO
    // dbg("Skip mergable Multi-Branch", in_graph);
    return false;
}

void NMutator::runSingleOpToNaiveMembound(Graph in_graph,
                                          std::vector<Graph> &out_graphs) {
    OpVec computeOps = in_graph->getComputeOps();
    assert(computeOps.size() == 1);
    const auto &computeOp = computeOps[0];
    auto g = infini::make_ref<GraphObj>(in_graph->getRuntime());
    nnet::Expr expr = opToExpression(computeOp);
    auto inputsN = nnet::GetTensorsVisitor().get(expr);
    // dbg(inputsN, expr);
    IT_ASSERT(inputsN.count("B") + inputsN.count("K") == 1,
              "Which one is the second input tensor?");
    vector<nnet::Tensor> inputsVectorN = {inputsN.at("A")};
    if (inputsN.count("B"))
        inputsVectorN.emplace_back(inputsN["B"]);
    else
        inputsVectorN.emplace_back(inputsN["K"]);
    // clone IF inputs and outputs into the new graph
    TensorVec inputsT, outputsT;
    for (auto t : computeOp->getInputs())
        inputsT.emplace_back(g->cloneTensor(t));
    for (auto t : computeOp->getOutputs())
        outputsT.emplace_back(g->cloneTensor(t));
    g->addOpWithOutputs<MemBoundObj>(inputsT, outputsT, inputsVectorN, expr,
                                     0.);
    g->print();
    out_graphs.emplace_back(g);
}

void NMutator::runSingleOp(Graph in_graph, std::vector<Graph> &out_graphs) {
    OpVec computeOps = in_graph->getComputeOps();
    IT_ASSERT(computeOps.size() == 1);
    if (Graph g = transformConvtransposed1x1(computeOps[0])) {
        out_graphs.emplace_back(g);
    }
    for (auto g : transformConv1x1(computeOps[0]))
        out_graphs.emplace_back(g);
    for (auto g : transformConv1xk(computeOps[0]))
        out_graphs.emplace_back(g);
    for (auto g : transformConv3x3ONNX(computeOps[0]))
        out_graphs.emplace_back(g);
    if (Graph g = transformG2bmm(computeOps[0])) {
        out_graphs.emplace_back(g);
    }
    if (Graph g = transformGbmm(computeOps[0])) {
        out_graphs.emplace_back(g);
    }
    if (infini::Graph g = transformDialtedConv(computeOps[0])) {
        out_graphs.emplace_back(g);
    }
    if (infini::Graph g = transformConvToGEMMReduce(computeOps[0])) {
        out_graphs.emplace_back(g);
    }
    if (infini::Graph g = transformConvTranposeToGEMMReduce(computeOps[0])) {
        out_graphs.emplace_back(g);
    }
    if (out_graphs.size() > 1)
        return;

    const set<OpType> opSet{OpType::Conv, OpType::ConvTransNHWC, OpType::G2BMM,
                            OpType::GBMM};
    if (opSet.count(computeOps[0]->getOpType()) == 0)
        return;

    auto expr = opToExpression(computeOps[0]);
    if (!expr)
        return;

    nnet::Derivator derivator(maxDepth);
    nnet::Formula conv_9x9(expr, 0);
    if (mode == Mode::Normal) {
        derivator.search(conv_9x9, 0);
    } else if (mode == Mode::RuleBased) {
        // dbg(derivationRules);
        derivator.ruleBasedDFS(conv_9x9, 0, derivationRules);
    } else
        IT_TODO_HALT_MSG("Unknown NMutator search mode.");
    const auto &candidates = derivator.getCandidates();
    // dbg(candidates.size());
    // derivator.print();
    for (const auto &candidate : candidates) {
        // dbg(nnet::FullPrinterVisitor().print(candidate.root));
        if (auto g = expressionToGraph(candidate.root, in_graph)) {
            out_graphs.emplace_back(g);
            hasTunedKernel = true;
        }
        // break; // HACK:Debug only for the first subgraph
    }
    // dbg(out_graphs);
    // for (auto graph : out_graphs) {
    //     graph->print();
    // }
    cntStates += derivator.getNumIntermediateStates();
    cntCandidates += derivator.getNumCandidates();
}

void NMutator::runMultipleOps(Graph in_graph, std::vector<Graph> &out_graphs) {
    IT_TODO_HALT();
    // std::cout << "run multiple ops" << std::endl;
    // in_graph->print();
    // std::cout << std::endl;

    // std::vector<Operator> computeOps;
    // dbg(computeOps);
    // in_graph->getComputeOps(computeOps);
    // nnet::VecExpr exprs;
    // for (const auto &op : computeOps)
    //     exprs.emplace_back(opToExpression(op));
    // dbg(exprs);

    // nnet::Derivator derivator;
    // nnet::MultiFormulas origin(exprs, 0);
    // bool canCombine = derivator.stageCombination(origin, 0);
    // dbg(canCombine);
    // const auto matmul0 = dynamic_cast<MatmulOp *>(computeOps[0]);
    // assert(matmul0);
    // // Build merged graph
    // auto g = new infini::Graph();
    // std::vector<Tensor *> inputsT, weightsT, outputsT;
    // for (const auto &opT : computeOps) {
    //     inputsT.emplace_back(opT->getInputs(0));
    //     weightsT.emplace_back(opT->getInputs(1));
    //     outputsT.emplace_back(opT->getOutput());
    // }
    // const auto concat1 = g->concat(inputsT, 0);
    // const auto concat2 = g->concat(weightsT, 0);
    // const auto matmul = g->matmul(concat1->getOutput(), concat2->getOutput(),
    //                               matmul0->getTransA(),
    //                               matmul0->getTransB());
    // g->split(matmul->getOutput(), outputsT, 0, computeOps.size());
    // // Build computation graph in PET:
    // g->updateConnection();
    // Graph graph = new Graph(g->getOperators());
    // out_graphs.emplace_back(graph);
    // // DEBUG
    // dbg(out_graphs);
    // for (auto graph : out_graphs) {
    //     graph->print();
    // }
}

// uint64_t NMutator::computeHashForSingleComputeOp(const Operator op) {
// if (op->getOpType() == OpType::Conv) {
//     auto conv = as<ConvObj>(op);
//     auto hash = conv->getHash();
//     auto inputDim = conv->getInputs()[0]->getDims();
//     auto weightDim = conv->getOutputs()[0]->getDims();
//     hash += inputDim[0] * 10000019 + inputDim[1] * 10000079 +
//             inputDim[2] * 10000103 + inputDim[3] * 10000121 +
//             weightDim[0] * 10000139 + weightDim[1] * 10000141 +
//             weightDim[2] * 10000169 + weightDim[3] * 10000189;
//     return hash;
// }
// else if (op->getOpType() == OpType::ConvTrans) {
//     auto conv = dynamic_cast<const ConvTransOp *>(op);
//     auto hash = conv->getHash();
//     auto inputDim = conv->getInputs()[0]->getDims();
//     auto weightDim = conv->getOutputs()[0]->getDims();
//     hash += inputDim[0] * 10000019 + inputDim[1] * 10000079 +
//             inputDim[2] * 10000103 + inputDim[3] * 10000121 +
//             weightDim[0] * 10000139 + weightDim[1] * 10000141 +
//             weightDim[2] * 10000169 + weightDim[3] * 10000189;
//     return hash;
// } else if (op->getOpType() == OpType::Matmul) {
//     static uint64_t matmulhash = 0;
//     return matmulhash++;
// } else if (op->getOpType() == OpType::G2BMM) {
//     auto g2bmm = dynamic_cast<const G2BMMOp *>(op);
//     auto hash = g2bmm->getHash();
//     auto inputDim = g2bmm->getInputs()[0]->getDims();
//     auto weightDim = g2bmm->getOutputs()[0]->getDims();
//     hash += inputDim[0] * 10000019 + inputDim[1] * 10000079 +
//             inputDim[2] * 10000103 + inputDim[3] * 10000121 +
//             weightDim[0] * 10000139 + weightDim[1] * 10000141 +
//             weightDim[2] * 10000169 + weightDim[3] * 10000189;
//     return hash;
// } else if (op->getType() == Operator::GBMML) {
//     auto gbmml = dynamic_cast<const GBMMLOp *>(op);
//     auto hash = gbmml->getHash();
//     auto inputDim = gbmml->getInputs()[0]->getDims();
//     auto weightDim = gbmml->getOutputs()[0]->getDims();
//     hash += inputDim[0] * 10000019 + inputDim[1] * 10000079 +
//             inputDim[2] * 10000103 + inputDim[3] * 10000121 +
//             weightDim[0] * 10000139 + weightDim[1] * 10000141 +
//             weightDim[2] * 10000169 + weightDim[3] * 10000189;
//     return hash;
// }
// else
//     {
//         IT_TODO_HALT();
//         return 0;
//     }
// }

nnet::Expr NMutator::opToExpression(Operator opT) {
    if (auto op = as<ConvObj>(opT)) {
        if (op->getSh() != 1 || op->getSw() != 1 || op->getDh() != 1 ||
            op->getDw() != 1)
            return nullptr;
    }
    auto [expr, mapNameNToTensorT] = extractOp(opT);
    IT_ASSERT(expr,
              "Cannot convert " + opT->toString() + " to an NNet expression");
    for (auto &[name, tensorT] : mapNameNToTensorT) {
        IT_ASSERT(inputsNameNToTensorT.count(name) == 0);
        inputsNameNToTensorT[name] = tensorT;
    }
    return expr;
}

pair<nnet::Expr, NMutator::NameNToTensorT> NMutator::extractOp(Operator opT) {
    if (auto convOp = as<ConvObj>(opT)) {
        const auto &inputs = convOp->getInputs();
        const auto &AT = inputs[0];
        const auto &KT = inputs[1];
        const auto &[n, c, h, w, f, r, s] = convOp->getNCHWFRS();
        const auto &[ph, pw, sh, sw, dh, dw] = convOp->getPadStrideDilation();
        if (!(sh == 1 && sw == 1 && dh == 1 && dw == 1))
            return {};
        assert(sh == 1 && sw == 1 && dh == 1 && dw == 1);
        const auto A = nnet::makeTensor("A", AT->getDims(),
                                        std::vector<int>{0, 0, ph, pw});
        const auto K = nnet::makeTensor("K", KT->getDims());
        return {nnet::ConvPattern::getExpr(A, K, n, c, h, w, f, r, s),
                {{"A", AT}, {"K", KT}}};
    } else if (auto convOp = as<ConvTransposed2dNHWCObj>(opT)) {
        const auto &AT = convOp->getInputs()[0];
        const auto &KT = convOp->getInputs()[1];
        const auto &[n, c, h, w, f, r, s] = convOp->getNCHWFRS();
        const auto &[ph, pw, sh, sw, dh, dw] = convOp->getPadStrideDilation();
        IT_ASSERT_TODO(convOp->getNumGroups() == 1);
        if (r != 4)
            return {};
        IT_ASSERT_TODO(r == 4);
        IT_ASSERT_TODO(ph == pw);
        IT_ASSERT_TODO(tie(sh, sw) == tuple(2, 2));
        IT_ASSERT_TODO(tie(dh, dw) == tuple(1, 1));

        // https://pytorch.org/docs/stable/generated/torch.nn.ConvTranspose2d.html
        // Real padding = dilation * (kernel_size - 1) - padding
        int padding = dh * (r - 1) - ph;
        const auto A = nnet::makeTensor(
            "A", AT->getDims(), std::vector<int>{0, padding, padding, 0});
        const auto K = nnet::makeTensor("K", KT->getDims());
        return {nnet::ConvTransPattern::getExpr(A, K, n, c, h, w, f, r, s),
                {{"A", AT}, {"K", KT}}};
    } else if (auto g2bmmOp = as<G2BMMObj>(opT)) {
        const auto &AT = g2bmmOp->getInputs()[0];
        const auto &BT = g2bmmOp->getInputs()[1];
        const auto [b, m, k, width, dilation] = g2bmmOp->getBMKWD();

        const auto &[expr, inputsN] =
            nnet::Sg2bmmPattern::getExpr(b, m, k, width, dilation);
        return {
            expr,
            {{inputsN.first->getName(), AT}, {inputsN.second->getName(), BT}}};
    } else if (auto gbmmlOp = as<GBMMObj>(opT)) {
        const auto &AT = gbmmlOp->getInputs()[0];
        const auto &BT = gbmmlOp->getInputs()[1];
        const auto [b, m, w, k, dilation] = gbmmlOp->getBMWND();
        const auto &[expr, inputsN] =
            nnet::LongformerGBMMPattern::getExpr(b, m, w, k, dilation);
        return {
            expr,
            {{inputsN.first->getName(), AT}, {inputsN.second->getName(), BT}}};
    } else if (auto matmulOp = as<MatmulObj>(opT)) {
        const auto &AT = matmulOp->getInputs()[0];
        const auto &BT = matmulOp->getInputs()[1];
        const auto [b, m, n, k, transA, transB] = matmulOp->getBMNKTransAB();
        const auto &[expr, inputsN] =
            nnet::MatmulPattern::getExpr(transA, transB, b, m, n, k);
        return {
            expr,
            {{inputsN.first->getName(), AT}, {inputsN.second->getName(), BT}}};
    } else if (auto op = as<MemBoundObj>(opT)) {
        NameNToTensorT m;
        for (int i = 0; i < op->numInputs(); ++i)
            m[op->getNnetInputs()[i]->getName()] = opT->getInputs()[i];
        return {op->getNnetExpr(), m};
    } else if (opT->getOpType() == OpType::Relu ||
               opT->getOpType() == OpType::Tanh) {
        return generateUnaryExpr(opT);
    }
    // // else if (auto transposeOp = dynamic_cast<TransposeOp *>(opT)) {
    // //     return transposeOpToExpression(transposeOp);
    // // }
    return {};
}

infini::Graph NMutator::expressionToGraph(nnet::Expr expr, Graph in_graph) {
    auto g = make_ref<GraphObj>(runtime);
    nnet::FullPrinterVisitor fullVisitor;
    // Get tensors in the reversed topological order
    const auto &tensorQueueN = fullVisitor.traverse(expr);
    // dbg(fullVisitor.print(expr));

    // Build a map: name in nnet -> tensors in infini
    // Add input tensors to the map
    std::map<std::string, Tensor> nameNToTensorT;
    for (const auto &[k, v] : inputsNameNToTensorT)
        nameNToTensorT[k] = g->cloneTensor(v);

    // Add output tensors to the map
    const auto &outputsT = in_graph->getOutputs();
    if (outputsT.size() != 1) {
        nnet_unimplemented_continue();
        return nullptr;
    }
    nameNToTensorT[std::get<0>(tensorQueueN.at(0))] =
        g->cloneTensor(outputsT[0]);
    // Skip the first tensor, which is output and should be created by clone
    for (size_t i = 1; i < tensorQueueN.size(); ++i) {
        const auto &[nameN, routineN, tensorN] = tensorQueueN[i];
        // dbg(nameN, routineN, tensorN);
        if (!routineN) {
            // this tensor is an input as it is not contrusted by a routine
            IT_ASSERT(nameNToTensorT.count(nameN),
                      "Missing an input tensor in graph or a rountine for this "
                      "tensor.");
        } else { // this tensor is an intermediate result
            IT_ASSERT(!nameNToTensorT.count(nameN),
                      "An NNET tensor appears twice or it is an input tensor "
                      "with routine specified.");
            nameNToTensorT[nameN] = g->addTensor(tensorN->getShape());
        }
    }

    // Build computation graph in InfiniTensor
    for (int i = tensorQueueN.size() - 1; i >= 0; --i) {
        const auto &[outputNameN, routineN, tensorN] = tensorQueueN[i];
        if (!routineN)
            continue;
        // dbg(outputNameN, routineN, tensorN, routineN->getType());
        if (auto op = nnet::as<nnet::ConvNode>(routineN)) {
            std::vector<nnet::Tensor> inputsN = op->getInputs();
            auto A = nameNToTensorT.at(inputsN[0]->getName());
            auto K = nameNToTensorT.at(inputsN[1]->getName());
            auto output = nameNToTensorT.at(outputNameN);
            const auto &[ph, pw, sh, sw, dh, dw] = op->getArgs();
            g->addOpWithOutputs<ConvObj>(A, K, output, ph, pw, sh, sw, dh, dw);
        } else if (auto op = nnet::as<nnet::ElementWiseNode>(routineN)) {
            // dbg(op, op->getExpr());
            // TODO: For a single input channel conv, it can be transformed into
            //          vec X vec ---> matrix --reduce--> result
            // This transformation only introduce membound Ops and can have a
            // wrong estimated execution time, so we skip it now.
            if (op->getInputs().size() != 1)
                return nullptr;
            nnet::MatchReshapeVisitor matchReshapeVisitor;
            // If this routine only change the shape, translate it to a Reshape
            if (matchReshapeVisitor(op->getExpr())) {
                auto input =
                    nameNToTensorT.at(op->getInputs().at(0)->getName());
                auto output = nameNToTensorT.at(outputNameN);
                if (input->size() != output->size())
                    return nullptr;
                g->addOpWithOutputs<ReshapeObj>(input, output,
                                                output->getDims());
            } else {
                TensorVec inputsPET;
                TensorVec outputsPET = {nameNToTensorT.at(outputNameN)};
                for (const auto &inputN : op->getInputs())
                    inputsPET.emplace_back(
                        nameNToTensorT.at(inputN->getName()));
                // Re-estimate time here.
                ssize_t cnt = 0;
                for (const auto &tensor : inputsPET)
                    cnt += tensor->size();
                for (const auto &tensor : outputsPET)
                    cnt += tensor->size();
                // dbg(inputsPET, outputsPET, op->getInputs(), op->getExpr(),
                //     memboundTime(cnt));
                g->addOpWithOutputs<MemBoundObj>(inputsPET, outputsPET,
                                                 op->getInputs(), op->getExpr(),
                                                 memboundTime(cnt));
            }
        } else if (auto op = nnet::as<nnet::MatmulNode>(routineN)) {
            assert(op->getInputs().size() == 2);
            nnet::Tensor AN = op->getInputs()[0];
            nnet::Tensor BN = op->getInputs()[1];
            TensorVec inputsPET = {nameNToTensorT.at(AN->getName()),
                                   nameNToTensorT.at(BN->getName())};
            TensorVec outputsPET = {nameNToTensorT.at(outputNameN)};
            const auto &[b, m, n, k, transa, transb] = op->getArgs();
            // // HACK: pruning for deubg
            if (!((transa == 0) && (transb == 1)))
                return nullptr;
            g->addOpWithOutputs<MatmulObj>(inputsPET[0], inputsPET[1],
                                           outputsPET[0], transa, transb);
        }
        // TODO
        // else if (auto op = nnet::as<nnet::G2bmmNode>(routineN)) {
        //     assert(op->getInputs().size() == 2);
        //     nnet::Tensor AN = op->getInputs()[0];
        //     nnet::Tensor BN = op->getInputs()[1];
        //     TensorVec inputsPET = {nameNToTensorT.at(AN->getName()),
        //                            nameNToTensorT.at(BN->getName())};
        //     TensorVec outputsPET = {nameNToTensorT.at(outputNameN)};
        //     const auto &[b, m, w, k, dilation] = op->getArgs();
        //     g->g2bmm(inputsPET[0], inputsPET[1], outputsPET[0], w, dilation);
        // } else if (auto op = nnet::as<nnet::GbmmNode>(routineN)) {
        //     assert(op->getInputs().size() == 2);
        //     nnet::Tensor AN = op->getInputs()[0];
        //     nnet::Tensor BN = op->getInputs()[1];
        //     TensorVec inputsPET = {nameNToTensorT.at(AN->getName()),
        //                            nameNToTensorT.at(BN->getName())};
        //     TensorVec outputsPET = {nameNToTensorT.at(outputNameN)};
        //     const auto &[b, m, w, n, dilation] = op->getArgs();
        //     g->gbmml(inputsPET[0], inputsPET[1], outputsPET[0], dilation);
        // }
        else
            IT_TODO_HALT();
    }
    return g;
}

double NMutator::memboundTime(ssize_t cnt) {
    return double(cnt) * 4 / bandwidth * 1000; // millisecond
}

double NMutator::memboundTime(const Shape &dims) {
    return memboundTime(dims.size());
}

Graph NMutator::transformDialtedConv(Operator _op) {
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

Graph NMutator::transformConvtransposed1x1(Operator _op) {
    auto op = as<ConvTransposed2dNHWCObj>(_op);
    if (!op)
        return nullptr;
    const auto &A = op->getInputs()[0];
    const auto &W = op->getInputs()[1];
    const auto &[n, c, h, w, f, r, s] = op->getNCHWFRS();
    const auto &[ph, pw, sh, sw, dh, dw] = op->getPadStrideDilation();
    const Shape inputDims = op->getInputs(0)->getDims();
    const Shape weightDims = op->getInputs(1)->getDims();
    const Shape outputDims = op->getOutput()->getDims();
    const DataType dtype = A->getDType();
    IT_ASSERT_TODO(op->getNumGroups() == 1);
    if (h != 1 || w != 1)
        return {};
    IT_ASSERT_TODO(ph == pw);
    if (tie(sh, sw) != tuple(1, 1)) {
        return nullptr;
    }
    IT_ASSERT_TODO(tie(dh, dw) == tuple(1, 1));
    auto g = make_ref<GraphObj>(runtime);
    // NHWF
    auto newA = g->addTensor(
        {inputDims[0] * inputDims[1] * inputDims[2], inputDims[3]}, dtype);
    // FRSC
    // auto newW = g->addTensor(
    //     {weightDims[0], weightDims[1] * weightDims[2] * weightDims[3]},
    //     dtype);
    // HACK: without transpoe
    auto newW = g->addTensor(
        {weightDims[1] * weightDims[2] * weightDims[3], weightDims[0]}, dtype);
    g->addOpWithOutputs<ReshapeObj>(g->cloneTensor(A), newA, newA->getDims());
    g->addOpWithOutputs<ReshapeObj>(g->cloneTensor(W), newW, newW->getDims());
    Tensor newO = g->addOp<MatmulObj>(newA, newW, nullptr, 0, 1)->getOutput();
    g->addOpWithOutputs<ReshapeObj>(newO, g->cloneTensor(op->getOutput()),
                                    op->getOutput()->getDims());
    return g;
}

Graph NMutator::transformConvToGEMMReduce(Operator _op) {
    auto op = as<ConvNHWCObj>(_op);
    if (!op)
        return nullptr;
    const auto &A = op->getInputs()[0];
    const auto &W = op->getInputs()[1];
    const auto &[n, c, h, w, f, r, s] = op->getNCHWFRS();
    const auto &[ph, pw, sh, sw, dh, dw] = op->getPadStrideDilation();
    const Shape inputDims = op->getInputs(0)->getDims();
    const Shape weightDims = op->getInputs(1)->getDims();
    const Shape outputDims = op->getOutput()->getDims();
    if (sh != 1 || sw != 1 || dh != 1 || dw != 1 ||
        (weightDims[2] != 1 && weightDims[3] != 1))
        return nullptr;
    const DataType dtype = A->getDType();
    auto g = make_ref<GraphObj>(runtime);
    auto newA = g->addTensor(
        {inputDims[0] * inputDims[1] * inputDims[2], inputDims[3]}, dtype);

    // // If use Matmul with transpose 0,0
    // auto newW = g->addTensor(
    //     {weightDims[3], weightDims[0] * weightDims[1] * weightDims[2]},
    //     dtype);

    // If use Matmul with transpose 0, 1
    auto newW = g->addTensor(
        {weightDims[0] * weightDims[1] * weightDims[2], weightDims[3]}, dtype);
    g->addOpWithOutputs<ReshapeObj>(g->cloneTensor(A), newA, newA->getDims());
    g->addOpWithOutputs<ReshapeObj>(g->cloneTensor(W), newW, newW->getDims());
    Tensor newO = g->addOp<MatmulObj>(newA, newW, nullptr, 0, 1)->getOutput();
    auto new1 = g->addTensor({n, h, w, f, r, s}, dtype);
    g->addOpWithOutputs<ReshapeObj>(newO, new1, new1->getDims());
    g->addOpWithOutputs<Conv2dReduce>(
        new1, nullptr, g->cloneTensor(op->getOutput()), false, 0.f, ph, pw);
    return g;
}

Graph NMutator::transformConvTranposeToGEMMReduce(Operator _op) {
    auto op = as<ConvTransposed2dNHWCObj>(_op);
    if (!op)
        return nullptr;
    const auto &A = op->getInputs()[0];
    const auto &W = op->getInputs()[1];
    // f is the de-facto input channel for ConvTranspose
    const auto &[n, c, h, w, f, r, s] = op->getNCHWFRS();
    const auto &[ph, pw, sh, sw, dh, dw] = op->getPadStrideDilation();
    if (dh != 1 || dw != 1)
        return nullptr;
    if (r == 4 && s == 4 && sh == 2 && sw == 2) // Solved by NNET
        return nullptr;
    const Shape inputDims = op->getInputs(0)->getDims();
    const Shape weightDims = op->getInputs(1)->getDims();
    const Shape outputDims = op->getOutput()->getDims();
    const DataType dtype = A->getDType();
    auto g = make_ref<GraphObj>(runtime);
    auto newA = g->addTensor( // [N,H,W,F]
        {inputDims[0] * inputDims[1] * inputDims[2], inputDims[3]}, dtype);
    auto newW = g->addTensor( // [F, CRS]
        {weightDims[0], weightDims[1] * weightDims[2] * weightDims[3]},
        dtype); // HACK: this should be a transpose

    g->addOpWithOutputs<ReshapeObj>(g->cloneTensor(A), newA, newA->getDims());
    g->addOpWithOutputs<ReshapeObj>(g->cloneTensor(W), newW, newW->getDims());
    // newO [NHW, CRS]
    Tensor newO = g->addOp<MatmulObj>(newA, newW, nullptr, 0, 0)->getOutput();
    auto new1 = g->addTensor({n, h, w, c, r, s}, dtype);
    g->addOpWithOutputs<ReshapeObj>(newO, new1, new1->getDims());
    // [NHW, CRS] -> [N,H,W,C]
    g->addOpWithOutputs<Conv2dReduceTranspose>(
        new1, nullptr, g->cloneTensor(op->getOutput()), false, 0.f, ph, pw, sh,
        sw, dh, dw);
    return g;
}

// Graph NMutator::transformConvtransposed(Operator _op) {
//     auto op = as<ConvTransposed2dNHWCObj>(_op);
//     if (!op)
//         return nullptr;
//     const auto &AT = op->getInputs()[0];
//     const auto &KT = op->getInputs()[1];
//     const auto &[n, c, h, w, f, r, s] = op->getNCHWFRS();
//     const auto &[ph, pw, sh, sw, dh, dw] = op->getPadStrideDilation();
//     IT_ASSERT_TODO(op->getNumGroups() == 1);
//     if (r != 4)
//         return {};
//     IT_ASSERT_TODO(ph == pw);
//     IT_ASSERT_TODO(tie(sh, sw) == tuple(2, 2));
//     IT_ASSERT_TODO(tie(dh, dw) == tuple(1, 1));

//     auto g = make_ref<Graph>();
//     // TODO: implement transformation rules
//     // How to efficiently write an expression...
//     auto inputDims = op->getInputs(0)->getDims();
//     auto weightDims = op->getInputs(1)->getDims();
//     auto outputDims = op->getOutput()->getDims();
//     // NHWF
//     auto newA =
//         g->tensor({inputDims[0] * inputDims[1] * inputDims[2],
//         inputDims[3]});
//     // RSFC
//     auto newW = g->tensor(
//         {weightDims[0] * weightDims[1] * weightDims[3], weightDims[2]});
//     auto newO = g->tensor({inputDims[0] * inputDims[1] * inputDims[2],
//                            weightDims[0] * weightDims[1] * weightDims[3]});
//     g->reshape(op->getInputs(0), newA);
//     g->reshape(op->getInputs(1), newW);
//     g->matmul(newA, newW, newO, 0, 1);
//     // g->reshape(newO, tconvOp->getOutput());
//     tconvOp->print();
//     dbg(newO->size() * 4, tconvOp->getOutput()->size() * 9);
//     assert(newO->size() * 4 == tconvOp->getOutput()->size() * 9);
//     g->membound({newO}, {tconvOp->getOutput()}, {}, nullptr,
//                 memboundTime(newO->size() + tconvOp->getOutput()->size()),
//                 "TConv3x3 reduce");
//     g->updateConnection();
//     Graph graph = new Graph(g->getOperators());
//     return graph;
// }

Graph NMutator::transformG2bmm(Operator _op) {
    auto op = as<G2BMMObj>(_op);
    if (!op || maxDepth <= 3)
        return nullptr;
    const auto [b, m, k, width, dilation] = op->getBMKWD();
    if (dilation == 1 || m % dilation != 0)
        return nullptr;
    auto g = make_ref<GraphObj>(runtime);
    auto A = g->cloneTensor(op->getInputs(0));
    auto B = g->cloneTensor(op->getInputs(1));
    auto O = g->cloneTensor(op->getOutput());
    auto A3 = splitTransposeMerge(g, A, 1, dilation),
         B3 = splitTransposeMerge(g, B, 1, dilation);
    auto O3 = g->addOp<G2BMMObj>(A3, B3, nullptr, width, 1)->getOutput();
    splitTransposeMerge(g, O3, 1, m / dilation, O);
    g->checkValid();
    return g;
}

Graph NMutator::transformGbmm(Operator _op) {
    auto op = as<GBMMObj>(_op);
    if (!op || maxDepth <= 3)
        return nullptr;
    const auto [b, m, width, k, dilation] = op->getBMWND();
    if (dilation == 1 || m % dilation != 0)
        return nullptr;
    auto g = make_ref<GraphObj>(runtime);
    auto A = g->cloneTensor(op->getInputs(0)); // [b,m,2w+1]
    auto B = g->cloneTensor(op->getInputs(1)); // [b,m,n]
    auto O = g->cloneTensor(op->getOutput());  // [b,m,n]
    auto A3 = splitTransposeMerge(g, A, 1, dilation),
         B3 = splitTransposeMerge(g, B, 1, dilation);
    auto O3 = g->addOp<GBMMObj>(A3, B3, nullptr, 1)->getOutput();
    splitTransposeMerge(g, O3, 1, m / dilation, O);
    g->checkValid();
    return g;
}

vector<Graph> NMutator::transformConv1x1(Operator _op) {
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

vector<Graph> NMutator::transformConv1xk(Operator _op) {
    vector<Graph> ret;
    auto op = as<ConvObj>(_op);
    if (!op)
        return {};
    const auto &[ph, pw, sh, sw, dh, dw] = op->getPadStrideDilation();
    if (sh != 1 || sw != 1 || dh != 1 || dw != 1)
        return {};
    const auto &[n, c, h, w, f, r, s] = op->getNCHWFRS();
    op->print();
    const auto &A = op->getInputs(0);
    const auto &W = op->getInputs(1);
    const auto &O = op->getOutput();
    const Shape &shapeA = A->getDims();
    const Shape &shapeW = W->getDims();
    if (shapeW[2] == 1 || shapeW[3] == 1) {
        {
            auto g = make_ref<GraphObj>(runtime);
            auto A0 = g->cloneTensor(A);
            auto W0 = g->cloneTensor(W); // [F, C, R, S]
            auto A1 =
                g->addOp<TransposeObj>(A0, nullptr, vector<int>{0, 2, 3, 1})
                    ->getOutput(); // [N, H, W, C]
            auto A2 =
                g->addOp<ReshapeObj>(
                     A1, nullptr,
                     vector<int>{shapeA[0] * shapeA[2] * shapeA[3], shapeA[1]})
                    ->getOutput(); // [N*H*W, C]
            // auto A2 =
            //     g->addTensor({shapeA[0] * shapeA[2] * shapeA[3], shapeA[1]});
            // dbg(A0, A2);
            // g->addOpWithOutputs<AnyObj>(vector{A0}, vector{A2},
            //                             string("FakeOp"), vector<int>{});
            auto W1 =
                g->addOp<TransposeObj>(W0, nullptr, vector<int>{0, 2, 3, 1})
                    ->getOutput(); // [F,R,S,C]
            auto W2 =
                g->addOp<ReshapeObj>(
                     W1, nullptr,
                     vector<int>{shapeW[2] * shapeW[3] * shapeW[0], shapeW[1]})
                    ->getOutput(); // [F*R*S, C]
            auto O0 =
                g->addOp<MatmulObj>(A2, W2, nullptr, 0, 1) // Original: W X A
                    ->getOutput();                         // [N*H*W, F*R*S]
            vector<int> args{op->getAct() != ActType::None,
                             n,
                             h,
                             w,
                             f,
                             r,
                             s,
                             O->getDims()[2],
                             O->getDims()[3],
                             ph,
                             pw,
                             sh,
                             sw,
                             dh,
                             dw};
            const string kernelName = "reduceConvRxSToNCHW";
            // const string kernelName = "FakeOp";
            auto O3 = g->addOpWithOutputs<AnyObj>(
                vector{O0}, vector{g->cloneTensor(O)}, kernelName, args);
            ret.emplace_back(g);
        }
        {
            auto g = make_ref<GraphObj>(runtime);
            auto A0 = g->cloneTensor(A);
            auto W0 = g->cloneTensor(W); // [F, C, R, S]
            auto A1 =
                g->addOp<TransposeObj>(A0, nullptr, vector<int>{0, 2, 3, 1})
                    ->getOutput(); // [N, H, W, C]
            auto A2 =
                g->addOp<ReshapeObj>(
                     A1, nullptr,
                     vector<int>{shapeA[0] * shapeA[2] * shapeA[3], shapeA[1]})
                    ->getOutput(); // [N*H*W, C]
            // auto A2 = // [N*H*W, C]
            //     g->addTensor({shapeA[0] * shapeA[2] * shapeA[3], shapeA[1]});
            // dbg(A0, A2);
            // g->addOpWithOutputs<AnyObj>(vector{A0}, vector{A2},
            //                             string("FakeOp"), vector<int>{});
            auto W1 =
                g->addOp<TransposeObj>(W0, nullptr, vector<int>{0, 2, 3, 1})
                    ->getOutput(); // [F,R,S,C]
            auto W2 =
                g->addOp<ReshapeObj>(
                     W1, nullptr,
                     vector<int>{shapeW[2] * shapeW[3] * shapeW[0], shapeW[1]})
                    ->getOutput(); // [F*R*S, C]
            auto O0 =
                g->addOp<MatmulObj>(W2, A2, nullptr, 0, 1) // Original: W X A
                    ->getOutput();                         // [F*R*S, N*H*W]
            vector<int> args{op->getAct() != ActType::None,
                             n,
                             h,
                             w,
                             f,
                             r,
                             s,
                             O->getDims()[2],
                             O->getDims()[3],
                             ph,
                             pw,
                             sh,
                             sw,
                             dh,
                             dw};
            // FIXME: FRS,NHW->NHWF
            const string kernelName = "reduceConvRxSToNCHW";
            // const string kernelName = "FakeOp";
            auto O3 = g->addOpWithOutputs<AnyObj>(
                vector{O0}, vector{g->cloneTensor(O)}, kernelName, args);
            ret.emplace_back(g);
        }
    }
    return ret;
}

Graph NMutator::constructGraphByOperatorChain(vector<Operator> ops,
                                              Graph inputGraph) {
    // Construct new graph
    auto g = make_ref<GraphObj>(runtime);
    IT_ASSERT(inputGraph->getInputs().size() == 1);
    IT_ASSERT(inputGraph->getOutputs().size() == 1);
    IT_ASSERT(ops.size() > 0,
              "TODO: If there is no op left, how to return an empty graph? " +
                  inputGraph->toString());
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
    if (g->getOutputs()[0]->getDims() != graphOutput->getDims())
        g->addOpWithOutputs<ReshapeObj>(input, graphOutput);
    return g;
}

Graph NMutator::eliminateVertically(const Graph &inputGraph) {
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

Graph NMutator::fuseVertically(const Graph &inputGraph) {
    Graph optGraph = make_ref<GraphObj>(runtime);

    auto chainOps = inputGraph->getOperators();
    IT_ASSERT(!chainOps.empty());
    for (auto &op : chainOps) {
        IT_ASSERT(op->isMemBoundOp());
        IT_ASSERT_TODO(op->getInputs().size() == 1);
        IT_ASSERT(op->getOutputs().size() == 1);
    }
    if (chainOps.size() == 1) {
        return make_ref<GraphObj>(runtime, chainOps);
    }
    std::vector<nnet::Expr> exprs;
    for (const auto &op : chainOps) {
        auto [expr, _] = extractOp(op);
        if (!expr)
            return nullptr;
        exprs.emplace_back(expr);
        // dbg(op, infini::as<nnet::RangeOpNode>(expr)->getFullExpression());
    }
    // double maxTime = getMaxPerf(std::make_shared<SubGraph>(chainOps));
    // Fuse a MemboundOp chain
    auto expr = nnet::MergeMemboundMutator(exprs).merge(true);
    auto inputNMap = nnet::GetTensorsVisitor().get(exprs.front());
    IT_ASSERT(inputNMap.size() == 1);
    vector<nnet::Tensor> inputsN;
    for (const auto &[name, t] : inputNMap) {
        inputsN.emplace_back(t);
    }
    optGraph->addOpWithOutputs<MemBoundObj>(chainOps.front()->getInputs(),
                                            chainOps.back()->getOutputs(),
                                            inputsN, expr, 0);
    // TODO: set time
    return optGraph;
}

pair<nnet::Expr, NMutator::NameNToTensorT>
NMutator::generateUnaryExpr(const Operator &op) {
    using namespace nnet;
    const map<OpType, nnet::FuncType> opTToFuncN = {
        {OpType::PRelu, nnet::FuncType::PRelu},
        {OpType::Relu, nnet::FuncType::Relu},
        {OpType::Tanh, nnet::FuncType::Tanh}};
    Shape shape = op->getInputs()[0]->getDims();
    nnet::FuncType type = opTToFuncN.at(op->getOpType());
    auto T = make_ref<TensorNode>("T", shape);
    VecExpr indices;
    for (size_t i = 0; i < shape.size(); ++i) {
        indices.emplace_back(make_ref<VarNode>("i" + std::to_string(i)));
    }
    auto sub = makeSubscript(T, indices);
    auto func = nnet::make_ref<FuncNode>(sub, type);
    vector<VarRangePair> varRanges;
    for (size_t i = 0; i < shape.size(); ++i) {
        varRanges.emplace_back(nnet::as<VarNode>(indices[i]),
                               Range{0, shape[i]});
    }
    return {makeRangeOperator(varRanges, {}, func),
            NameNToTensorT{{"T", op->getInputs()[0]}}};
}

pair<nnet::Expr, vector<nnet::Tensor>> NMutator::generateRevert(Tensor in) {
    using namespace nnet;
    using infini::make_ref;
    const Shape &orignalShape = in->getDims();
    auto tensor = makeTensor("T", in->getDims());
    VecExpr iters;
    for (size_t i = 0; i < orignalShape.size(); ++i) {
        iters.emplace_back(make_ref<VarNode>("i" + std::to_string(i)));
    }

    Shape newShape = orignalShape;
    std::reverse(newShape.begin(), newShape.end());
    auto sub = makeSubscript(tensor, iters);
    vector<VarRangePair> loopIters;
    for (int i = orignalShape.size() - 1; i >= 0; --i) {
        loopIters.emplace_back(infini::as<VarNode>(iters[i]),
                               Range{0, orignalShape[i]});
    }
    auto range = makeRangeOperator(loopIters, {}, sub);
    return {range, {tensor}};
}

Tensor NMutator::splitTransposeMerge(Graph g, Tensor A, int dim, int chunkSize,
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

vector<Graph> NMutator::transformConv3x3ONNX(Operator _op) {
    vector<Graph> ret;
    auto op = as<ConvObj>(_op);
    if (!op)
        return ret;
    const auto &[n, c, h, w, f, r, s] = op->getNCHWFRS();
    const auto &[ph, pw, sh, sw, dh, dw] = op->getPadStrideDilation();
    if (tuple{n, c, h, w, f, r, s} != tuple{1, 512, 7, 7, 512, 3, 3} ||
        tuple{ph, pw, sh, sw, dh, dw} != tuple{1, 1, 1, 1, 1, 1})
        return ret;
    auto g = make_ref<GraphObj>(runtime);
    auto A = g->cloneTensor(op->getInputs(0));
    auto W = g->cloneTensor(op->getInputs(1)); // [F, C, R, S]
    auto O = g->cloneTensor(op->getOutput());
    A = g->addOp<ReshapeObj>(A, nullptr, vector<int>{c, h * w})
            ->getOutput(); // [C, H*W]
    W = g->addOp<ReshapeObj>(W, nullptr, vector<int>{f * r * s, c})
            ->getOutput();                             // [F,R,S,C]
    auto O0 = g->addOp<MatmulObj>(W, A, nullptr, 0, 0) // Orignal: W X A
                  ->getOutput();                       // [F*R*S, H*W]
    vector<int> args{};
    const string kernelName = "Reduce3x3Offset_hint";
    // const string kernelName = "FakeOp";
    auto O3 = g->addOpWithOutputs<AnyObj>(vector{O0}, vector{g->cloneTensor(O)},
                                          kernelName, args);
    hasTunedKernel = true; // enforce the transformation
    ret.emplace_back(g);
    return ret;
}

} // namespace infini
