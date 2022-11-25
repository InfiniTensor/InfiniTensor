#include "nnet/nmutator.h"
#include "core/graph.h"
#include "nnet/Visitor/FullPrinterVisitor.h"
#include "nnet/Visitor/GetTensorsVisitor.h"
#include "nnet/Visitor/MatchReshapeVisitor.h"
#include "nnet/derivator.h"
#include "operators/conv.h"
#include "operators/matmul.h"
#include "operators/membound.h"

namespace infini {

NMutator::NMutator(Mode mode) : Mutator(10), mode{mode} {
    IT_ASSERT(mode != Mode::RuleBased, "Use RuleBased in the other ctor.");
}

NMutator::NMutator(const std::vector<int> &derivationRules)
    : Mutator(10), mode{Mode::RuleBased}, derivationRules{derivationRules} {}

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

void NMutator::runSingleOpToNaiveMembound(Graph in_graph,
                                          std::vector<Graph> &out_graphs) {
    OpVec computeOps = in_graph->getComputeOps();
    assert(computeOps.size() == 1);
    const auto &computeOp = computeOps[0];
    auto g = infini::make_ref<GraphObj>(in_graph->getRuntime());
    auto expr = opToExpression(computeOp);
    auto inputsN = nnet::GetTensorsVisitor().get(expr);
    dbg(inputsN, expr);
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
    IT_TODO_HALT();
    // OpVec computeOps = in_graph->getComputeOps();
    // if (infini::Graph g = transformTConv1x1(computeOps[0])) {
    //     out_graphs.emplace_back(g);
    //     return;
    // }
    // // Commented for debug, not implemented yet
    // // if (infini::Graph g = transformTConv3x3(computeOps[0])) {
    // //     Graph graph = new Graph(g->getOperators());
    // //     out_graphs.emplace_back(graph);
    // //     return;
    // // }
    // if (infini::Graph g = transformDialtedConv(computeOps[0])) {
    //     out_graphs.emplace_back(g);
    //     return;
    // }
    // // if (infini::Graph g = transformConv1x1(computeOps[0])) {
    // //     Graph graph = new Graph(g->getOperators());
    // //     out_graphs.emplace_back(graph);
    // //     return;
    // // }
    // // if (infini::Graph g = transformConv1xk(computeOps[0])) {
    // //     Graph graph = new Graph(g->getOperators());
    // //     out_graphs.emplace_back(graph);
    // //     return;
    // // }

    // auto expr = opToExpression(computeOps[0]);
    // if (!expr)
    //     return;

    // nnet::Derivator derivator(maxDepth);
    // nnet::Formula conv_9x9(expr, 0);
    // // const std::vector<int> rules{3, 2, 2, 2, 2, 5, 8, 8, 6, 91, 90}; //
    // Tconv
    // // const std::vector<int> rules{1, 7, 7, 2, 8, 6, 6}; // G2BMM
    // if (mode == Mode::Normal) {
    //     derivator.search(conv_9x9, 0);
    // } else if (mode == Mode::RuleBased) {
    //     dbg(derivationRules);
    //     derivator.ruleBasedDFS(conv_9x9, 0, derivationRules);
    // } else
    //     nnet_assert(0, "Unknown mode");
    // const auto &candidates = derivator.getCandidates();
    // dbg(candidates.size());
    // // derivator.print();
    // for (const auto &candidate : candidates) {
    //     // dbg(nnet::FullPrinterVisitor().print(candidate.root));
    //     if (auto g = expressionToGraph(candidate.root, in_graph)) {
    //         out_graphs.emplace_back(g);
    //     }
    //     // break; // HACK:Debug only for the first subgraph
    // }
    // // dbg(out_graphs);
    // // for (auto graph : out_graphs) {
    // //     graph->print();
    // // }
    // cntStates += derivator.getNumIntermediateStates();
    // cntCandidates += derivator.getNumCandidates();
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

nnet::Expr NMutator::opToExpression(Operator op) {
    // IT_TODO_HALT();
    if (auto convOp = as<ConvObj>(op)) {
        const auto &inputs = convOp->getInputs();
        const auto &AT = inputs[0];
        const auto &KT = inputs[1];
        const auto &[n, c, h, w, f, r, s] = convOp->getNCHWFRS();
        const auto &[ph, pw, sh, sw, dh, dw] = convOp->getPadStrideDilation();
        if (!(sh == 1 && sw == 1 && dh == 1 && dw == 1))
            return nullptr;
        assert(sh == 1 && sw == 1 && dh == 1 && dw == 1);
        inputsNameNToTensorT["A"] = AT;
        inputsNameNToTensorT["K"] = KT;
        const auto A = nnet::makeTensor("A", AT->getDims(),
                                        std::vector<int>{0, 0, ph, pw});
        const auto K = nnet::makeTensor("K", KT->getDims());
        return nnet::ConvPattern::getExpr(A, K, n, c, h, w, f, r, s);
    } else if (auto convOp = as<ConvTransposed2dObj>(op)) {
        const auto &AT = convOp->getInputs()[0];
        const auto &KT = convOp->getInputs()[1];
        inputsNameNToTensorT["A"] = AT;
        inputsNameNToTensorT["K"] = KT;
        const auto &[n, c, h, w, f, r, s] = convOp->getNCHWFRS();
        const auto &[ph, pw, sh, sw, dh, dw] = convOp->getPadStrideDilation();
        IT_ASSERT_TODO(convOp->getNumGroups() == 1);
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
        return nnet::ConvTransPattern::getExpr(A, K, n, c, h, w, f, r, s);
        // } else if (auto g2bmmOp = dynamic_cast<G2BMMOp *>(op)) {
        //     const auto &AT = g2bmmOp->getInputs()[0];
        //     const auto &BT = g2bmmOp->getInputs()[1];
        //     const auto [b, m, k, width, dilation] = g2bmmOp->getArgs();

        //     const auto &[expr, inputsN] =
        //         nnet::Sg2bmmPattern::getExpr(b, m, k, width, dilation);
        //     inputsNameNToTensorT[inputsN.first->getName()] = AT;
        //     inputsNameNToTensorT[inputsN.second->getName()] = BT;
        //     return expr;
        // } else if (auto gbmmlOp = dynamic_cast<GBMMLOp *>(op)) {
        //     const auto &AT = gbmmlOp->getInputs()[0];
        //     const auto &BT = gbmmlOp->getInputs()[1];
        //     const auto [b, m, w, k, dilation] = gbmmlOp->getArgs();
        //     const auto &[expr, inputsN] =
        //         nnet::LongformerGBMMPattern::getExpr(b, m, w, k, dilation);
        //     inputsNameNToTensorT[inputsN.first->getName()] = AT;
        //     inputsNameNToTensorT[inputsN.second->getName()] = BT;
        //     dbg(b, m, w, k, dilation, expr);
        //     return expr;
    } else if (auto matmulOp = as<MatmulObj>(op)) {
        const auto &AT = matmulOp->getInputs()[0];
        const auto &BT = matmulOp->getInputs()[1];
        const auto [b, m, n, k, transA, transB] = matmulOp->getBMNKTransAB();
        const auto &[expr, inputsN] =
            nnet::MatmulPattern::getExpr(transA, transB, b, m, n, k);
        inputsNameNToTensorT[inputsN.first->getName()] = AT;
        inputsNameNToTensorT[inputsN.second->getName()] = BT;
        // dbg(b, m, n, k, expr);
        return expr;
    }
    // // else if (auto transposeOp = dynamic_cast<TransposeOp *>(op)) {
    // //     return transposeOpToExpression(transposeOp);
    // // }
    nnet_unimplemented_continue();
    return nullptr;
}

infini::Graph NMutator::expressionToGraph(nnet::Expr expr, Graph in_graph) {
    IT_TODO_HALT();
    // auto g = make_ref<GraphObj>();
    // nnet::FullPrinterVisitor fullVisitor;
    // const auto &tensorQueueN = fullVisitor.traverse(expr);
    // // Build tensors: Skip the first one, which is output
    // auto nameNToTensorT = inputsNameNToTensorT;
    // for (size_t i = 1; i < tensorQueueN.size(); ++i) {
    //     const auto &[nameN, routineN, tensorN] = tensorQueueN[i];
    //     // dbg(nameN, routineN, tensorN);
    //     if (!routineN) {
    //         // This is an inputs
    //         assert(nameNToTensorT.count(nameN));
    //     } else {
    //         assert(!nameNToTensorT.count(nameN));
    //         nameNToTensorT[nameN] = g->addTensor(tensorN->getShape());
    //     }
    // }
    // const auto &outputsPET = in_graph->getOutputs();
    // if (outputsPET.size() != 1) {
    //     nnet_unimplemented_continue();
    //     return nullptr;
    // }
    // nameNToTensorT[std::get<0>(tensorQueueN.at(0))] = outputsPET[0];
    // // Build computation graph in PET:
    // for (int i = tensorQueueN.size() - 1; i >= 0; --i) {
    //     const auto &[outputNameN, routineN, tensorN] = tensorQueueN[i];
    //     if (!routineN)
    //         continue;
    //     // dbg(outputNameN, routineN, tensorN, routineN->getType());
    //     if (auto op = nnet::as<nnet::ConvNode>(routineN)) {
    //         // g->conv(i8, w9, 2, 2);
    //         std::vector<nnet::Tensor> inputsN = op->getInputs();
    //         auto A = nameNToTensorT.at(inputsN[0]->getName());
    //         auto K = nameNToTensorT.at(inputsN[1]->getName());
    //         auto output = nameNToTensorT.at(outputNameN);
    //         const auto &[ph, pw, sh, sw, dh, dw] = op->getArgs();
    //         g->conv(A, K, output, ph, pw, sh, sw, dh, dw);
    //     } else if (auto op = nnet::as<nnet::ElementWiseNode>(routineN)) {
    //         assert(op->getInputs().size() == 1);
    //         nnet::MatchReshapeVisitor matchReshapeVisitor;
    //         if (matchReshapeVisitor(op->getExpr())) {
    //             auto input =
    //                 nameNToTensorT.at(op->getInputs().at(0)->getName());
    //             auto output = nameNToTensorT.at(outputNameN);
    //             g->reshape(input, output);
    //         } else {
    //             TensorVec inputsPET;
    //             TensorVec outputsPET = {nameNToTensorT.at(outputNameN)};
    //             for (const auto &inputN : op->getInputs())
    //                 inputsPET.emplace_back(
    //                     nameNToTensorT.at(inputN->getName()));
    //             // Re-estimate time here.
    //             ssize_t cnt = 0;
    //             for (const auto tensor : inputsPET)
    //                 cnt += tensor->size();
    //             for (const auto tensor : outputsPET)
    //                 cnt += tensor->size();
    //             g->membound(inputsPET, outputsPET, op->getInputs(),
    //                         op->getExpr(), memboundTime(cnt));
    //         }
    //     } else if (auto op = nnet::as<nnet::MatmulNode>(routineN)) {
    //         assert(op->getInputs().size() == 2);
    //         nnet::Tensor AN = op->getInputs()[0];
    //         nnet::Tensor BN = op->getInputs()[1];
    //         TensorVec inputsPET = {nameNToTensorT.at(AN->getName()),
    //                                nameNToTensorT.at(BN->getName())};
    //         TensorVec outputsPET = {nameNToTensorT.at(outputNameN)};
    //         const auto &[b, m, n, k, transa, transb] = op->getArgs();
    //         g->matmul(inputsPET[0], inputsPET[1], outputsPET[0], transa,
    //                   transb);
    //     } else if (auto op = nnet::as<nnet::G2bmmNode>(routineN)) {
    //         assert(op->getInputs().size() == 2);
    //         nnet::Tensor AN = op->getInputs()[0];
    //         nnet::Tensor BN = op->getInputs()[1];
    //         TensorVec inputsPET = {nameNToTensorT.at(AN->getName()),
    //                                nameNToTensorT.at(BN->getName())};
    //         TensorVec outputsPET = {nameNToTensorT.at(outputNameN)};
    //         const auto &[b, m, w, k, dilation] = op->getArgs();
    //         g->g2bmm(inputsPET[0], inputsPET[1], outputsPET[0], w, dilation);
    //     } else if (auto op = nnet::as<nnet::GbmmNode>(routineN)) {
    //         assert(op->getInputs().size() == 2);
    //         nnet::Tensor AN = op->getInputs()[0];
    //         nnet::Tensor BN = op->getInputs()[1];
    //         TensorVec inputsPET = {nameNToTensorT.at(AN->getName()),
    //                                nameNToTensorT.at(BN->getName())};
    //         TensorVec outputsPET = {nameNToTensorT.at(outputNameN)};
    //         const auto &[b, m, w, n, dilation] = op->getArgs();
    //         g->gbmml(inputsPET[0], inputsPET[1], outputsPET[0], dilation);
    //     }
    // }
    // g->updateConnection();
    // Graph graph = new Graph(g->getOperators());
    // return graph;
}

double NMutator::memboundTime(ssize_t cnt) {
    return double(cnt) * 4 / bandwidth * 1000; // millisecond
}

double NMutator::memboundTime(const Shape &dims) {
    return memboundTime(dims.size());
}

// infini::Graph NMutator::fuseHetConv(nnet::Expr expr, Graph in_graph) {
//     // Conv3x3+Conv1x1 => Gemm(nhw, f(rs+1), c) + Reduce
//     auto g = std::make_shared<infini::Graph>();
//     in_graph->print();
//     assert(in_graph->getInputs().size() == 3);
//     auto input = in_graph->getOperators()[0]->getInputs(0);
//     auto conv = dynamic_cast<ConvOp *>(in_graph->getOperators()[0]);
//     auto output = conv->getOutput();
//     // auto input = g->reshape(input);
//     auto inputTrans = g->transpose(input, 0, {-1, {0, 2, 3}, 1}, -1);
//     // dbg(inputTrans->getOutput()->getDims());
//     const auto &[n, c, h, w, f, r, s, ph, pw, sh, sw, dh, dw, G, bi, ac] =
//         conv->getArgs(0);
//     auto weight = g->tensor({1, c, f * (3 * 3 + 1)});
//     dbg(weight->getDims());
//     auto matmul = g->matmul(inputTrans->getOutput(), weight, false, false);
//     auto bias = g->tensor({f});
//     const double size = n * f * h * w * (3 * 3 + 1) * 4;
//     // FIXME: add NNET tensors for verfication
//     auto membound =
//         g->membound({matmul->getOutput(), bias}, {output}, {}, nullptr,
//                     memboundTime(size), "Reduce_conv3x3+1x1");
//     dbg(n, f, h, w);
//     dynamic_cast<MemBoundOp *>(membound)->setNFHW(n, f, h, w);

//     return new Graph(g->getOperators());
// }

// Graph NMutator::transformDialtedConv(Operator op) {
//     if (auto convOp = dynamic_cast<ConvOp *>(op)) {
//         if (convOp->getPh() == convOp->getDh() && convOp->getSh() == 1 &&
//             convOp->getDh() > 1 && convOp->getDh() == convOp->getDw()) {
//             const int d = convOp->getDh();
//             assert(convOp->getInputs()[0]->getDims()[2] % d == 0);
//             auto g = new infini::Graph();
//             auto inputDims = convOp->getInputs(0)->getDims();
//             auto weightDims = convOp->getInputs(1)->getDims();
//             auto outputDims = convOp->getOutput()->getDims();
//             auto newA = g->tensor({inputDims[0] * d * d, inputDims[1],
//                                    inputDims[2] / d, inputDims[3] / d});
//             // auto newW = g->tensor(
//             //     {weightDims[0] * weightDims[1] * weightDims[3],
//             //     weightDims[2]});
//             auto newO =
//                 g->tensor({inputDims[0] * inputDims[1] * inputDims[2],
//                            weightDims[0] * weightDims[1] * weightDims[3]});
//             g->membound(
//                 {convOp->getInputs(0)}, {newA}, {}, nullptr,
//                 memboundTime(convOp->getInputs(0)->size() + newA->size()),
//                 "DConv Tranpose Input");
//             // g->membound({convOp->getInputs(1)}, {newW}, {}, nullptr, 0,
//             //             "Tranpose Weight");
//             g->conv(newA, convOp->getInputs(1), newO, 1, 1, 1, 1);
//             g->reshape(newO, convOp->getOutput());
//             dbg(newO->size(), convOp->getOutput()->size());
//             assert(newO->size() == convOp->getOutput()->size());
//             g->membound(
//                 {newO}, {convOp->getOutput()}, {}, nullptr,
//                 memboundTime(newO->size() + convOp->getOutput()->size()),
//                 "DConv Tranpose Output");
//             g->updateConnection();
//             Graph graph = new Graph(g->getOperators());
//             return graph;
//         }
//     }
//     return nullptr;
// }

// Graph NMutator::transformTConv3x3(Operator op) {
//     if (auto tconvOp = dynamic_cast<ConvTransOp *>(op)) {
//         dbg(tconvOp->getInputs()[1]->getDims());
//         if (tconvOp->getPh() == 1 && tconvOp->getSh() == 2 &&
//             tconvOp->getInputs()[1]->getDims()[0] == 3 &&
//             tconvOp->getInputs()[1]->getDims()[1] == 3) {
//             auto g = new infini::Graph();
//             auto inputDims = tconvOp->getInputs(0)->getDims();
//             auto weightDims = tconvOp->getInputs(1)->getDims();
//             auto outputDims = tconvOp->getOutput()->getDims();
//             // NHWF
//             auto newA = g->tensor(
//                 {inputDims[0] * inputDims[1] * inputDims[2], inputDims[3]});
//             // RSFC
//             auto newW = g->tensor(
//                 {weightDims[0] * weightDims[1] * weightDims[3],
//                 weightDims[2]});
//             auto newO =
//                 g->tensor({inputDims[0] * inputDims[1] * inputDims[2],
//                            weightDims[0] * weightDims[1] * weightDims[3]});
//             g->reshape(tconvOp->getInputs(0), newA);
//             g->reshape(tconvOp->getInputs(1), newW);
//             g->matmul(newA, newW, newO, 0, 1);
//             // g->reshape(newO, tconvOp->getOutput());
//             tconvOp->print();
//             dbg(newO->size() * 4, tconvOp->getOutput()->size() * 9);
//             assert(newO->size() * 4 == tconvOp->getOutput()->size() * 9);
//             g->membound(
//                 {newO}, {tconvOp->getOutput()}, {}, nullptr,
//                 memboundTime(newO->size() + tconvOp->getOutput()->size()),
//                 "TConv3x3 reduce");
//             g->updateConnection();
//             Graph graph = new Graph(g->getOperators());
//             return graph;
//         }
//     }
//     return nullptr;
// }

// Graph NMutator::transformTConv1x1(Operator op) {
//     if (auto tconvOp = dynamic_cast<ConvTransOp *>(op)) {
//         if (tconvOp->getPh() == 0 && tconvOp->getSh() == 1) {
//             auto g = new infini::Graph();
//             auto inputDims = tconvOp->getInputs(0)->getDims();
//             auto weightDims = tconvOp->getInputs(1)->getDims();
//             auto outputDims = tconvOp->getOutput()->getDims();
//             auto newA = g->tensor(
//                 {inputDims[0] * inputDims[1] * inputDims[2], inputDims[3]});
//             auto newW = g->tensor(
//                 {weightDims[0] * weightDims[1] * weightDims[3],
//                 weightDims[2]});
//             auto newO =
//                 g->tensor({inputDims[0] * inputDims[1] * inputDims[2],
//                            weightDims[0] * weightDims[1] * weightDims[3]});
//             g->reshape(tconvOp->getInputs(0), newA);
//             g->reshape(tconvOp->getInputs(1), newW);
//             g->matmul(newA, newW, newO, 0, 1);
//             g->reshape(newO, tconvOp->getOutput());
//             g->updateConnection();
//             Graph graph = new Graph(g->getOperators());
//             return graph;
//         }
//     }
//     return nullptr;
// }

// Graph NMutator::transformConv1x1(Operator op) {
//     auto convOp = dynamic_cast<ConvOp *>(op);
//     if (!convOp)
//         return nullptr;
//     if (convOp->getPh() == 0 && convOp->getSh() == 1 &&
//         convOp->getInputs()[1]->getDims()[2] == 1 &&
//         convOp->getInputs()[1]->getDims()[3] == 1) {
//         // Transpose is requrired for BS>1
//         // if (convOp->getInputs()[0]->getDims()[0] == 1) {
//         auto g = new infini::Graph();
//         auto inputDims = convOp->getInputs(0)->getDims();
//         auto weightDims = convOp->getInputs(1)->getDims();
//         auto outputDims = convOp->getOutput()->getDims();
//         auto newA = g->tensor(
//             {inputDims[1], inputDims[0] * inputDims[2] * inputDims[3]});
//         auto newW = g->tensor({weightDims[0], weightDims[1]});
//         auto newO = g->tensor(
//             {weightDims[0], inputDims[0] * inputDims[2] * inputDims[3]});
//         g->reshape(convOp->getInputs(0), newA);
//         g->reshape(convOp->getInputs(1), newW);
//         g->matmul(newW, newA, newO, 0, 0);
//         g->reshape(newO, convOp->getOutput());
//         g->updateConnection();
//         Graph graph = new Graph(g->getOperators());
//         return graph;
//     }
//     return nullptr;
// }

// Graph NMutator::transformConv1xk(Operator op) {
//     auto convOp = dynamic_cast<ConvOp *>(op);
//     if (!convOp)
//         return nullptr;
//     if (convOp->getSh() != 1 || convOp->getSw() != 1)
//         return nullptr;
//     bool a = convOp->getInputs()[1]->getDims()[2] == 1;
//     bool b = convOp->getInputs()[1]->getDims()[3] == 1;
//     if (!(a ^ b))
//         return nullptr;
//     convOp->print();
//     auto g = new infini::Graph();
//     auto inputDims = convOp->getInputs(0)->getDims();
//     auto weightDims = convOp->getInputs(1)->getDims();
//     auto outputDims = convOp->getOutput()->getDims();
//     auto newA =
//         g->tensor({inputDims[0] * inputDims[2] * inputDims[3],
//         inputDims[1]});
//     auto newW = g->tensor(
//         {weightDims[0] * weightDims[2] * weightDims[3], weightDims[1]});
//     auto newO = g->tensor({weightDims[0] * weightDims[2] * weightDims[3],
//                            inputDims[0] * inputDims[2] * inputDims[3]});
//     // g->reshape(convOp->getInputs(0), newA);
//     g->membound({convOp->getInputs(0)}, {newA}, {}, nullptr,
//                 memboundTime(convOp->getInputs(0)->size() + newA->size()),
//                 "1xk input reshape");
//     g->reshape(convOp->getInputs(1), newW);

//     g->matmul(newW, newA, newO, 0, 1);
//     g->membound({newO}, {convOp->getOutput()}, {}, nullptr,
//                 memboundTime(newW->size() + convOp->getOutput()->size()),
//                 "1xk reduce");
//     g->updateConnection();
//     Graph graph = new Graph(g->getOperators());
//     return graph;
// }

} // namespace infini
