#include "code_gen/nnet/nmutator.h"
#include "code_gen/nnet/Visitor/FullPrinterVisitor.h"
#include "code_gen/nnet/Visitor/GetTensorsVisitor.h"
#include "code_gen/nnet/derivator.h"

namespace tpm {

NMutator::NMutator() {}

NMutator::~NMutator() {}

void NMutator::setToNaiveMembound() { mode = Mode::ToNaiveMembound; }

void NMutator::run(SubGraph *in_graph, std::vector<SubGraph *> &out_graphs,
                   int mdepth,
                   std::vector<std::shared_ptr<Operator>> candidate_ops,
                   float threshold) {
    // Test helper: naively transform one Op to Membound
    if (mode == Mode::ToNaiveMembound) {
        runSingleOpToNaiveMembound(in_graph, out_graphs);
        dbg(out_graphs.size());
        return;
    }
    // // Hack for HetConv fusion
    // if (statGraph(in_graph) == NMutator::SGType::HetConv) {
    //     dbg("Start fuse HetConv");
    //     out_graphs.emplace_back(fuseHetConv(nullptr, in_graph));
    // }
    // Clear input names maps with tensor
    inputsNameNToTensorT.clear();
    std::vector<Operator *> computeOps;
    in_graph->getComputeOps(computeOps);
    // assert(computeOps.size() == 1);
    if (computeOps.size() == 1)
        runSingleOp(in_graph, out_graphs);
    // FIXME: runMultipleOps results in segfault
    // else
    //     runMultipleOps(in_graph, out_graphs);
}

void NMutator::runSingleOpToNaiveMembound(SubGraph *in_graph,
                                          std::vector<SubGraph *> &out_graphs) {
    std::vector<Operator *> computeOps;
    in_graph->getComputeOps(computeOps);
    assert(computeOps.size() == 1);
    const auto &computeOp = computeOps[0];
    auto g = std::make_shared<tpm::Graph>();
    auto expr = opToExpression(computeOp);
    auto inputsN = nnet::GetTensorsVisitor().get(expr);
    dbg(inputsN);
    g->membound(computeOp->getInputs(), computeOp->getOutputs(),
                {inputsN.at("A"), inputsN.at("K")}, expr, 0);
    auto subgraph = new SubGraph(g->getOperators());
    subgraph->print();
    out_graphs.emplace_back(subgraph);
}

void NMutator::runSingleOp(SubGraph *in_graph,
                           std::vector<SubGraph *> &out_graphs) {
    std::vector<Operator *> computeOps;
    in_graph->getComputeOps(computeOps);
    if (tpm::SubGraph *g = transformTConv1x1(computeOps[0])) {
        SubGraph *graph = new SubGraph(g->getOperators());
        out_graphs.emplace_back(graph);
        return;
    }
    if (tpm::SubGraph *g = transformTConv3x3(computeOps[0])) {
        SubGraph *graph = new SubGraph(g->getOperators());
        out_graphs.emplace_back(graph);
        return;
    }
    if (tpm::SubGraph *g = transformDialtedConv(computeOps[0])) {
        SubGraph *graph = new SubGraph(g->getOperators());
        out_graphs.emplace_back(graph);
        return;
    }
    if (tpm::SubGraph *g = transformConv1x1(computeOps[0])) {
        SubGraph *graph = new SubGraph(g->getOperators());
        out_graphs.emplace_back(graph);
        return;
    }
    if (tpm::SubGraph *g = transformConv1xk(computeOps[0])) {
        SubGraph *graph = new SubGraph(g->getOperators());
        out_graphs.emplace_back(graph);
        return;
    }

    auto expr = opToExpression(computeOps[0]);
    dbg(expr);
    if (!expr)
        return;

    nnet::Derivator derivator(maxDepth);
    nnet::Formula conv_9x9(expr, 0);
    const std::vector<int> rules{3, 2, 2, 2, 2, 5, 8, 8, 6, 91, 90}; // Tconv
    // const std::vector<int> rules{1, 7, 7, 2, 8, 6, 6}; // G2BMM
    derivator.ruleBasedDFS(conv_9x9, 0, rules);
    // derivator.search(conv_9x9, 0);
    const auto &candidates = derivator.getCandidates();
    dbg(candidates.size());
    // derivator.print();
    for (const auto &candidate : candidates) {
        // dbg(nnet::FullPrinterVisitor().print(candidate.root));
        if (auto g = expressionToGraph(candidate.root, in_graph)) {
            SubGraph *graph = new SubGraph(g->getOperators());
            out_graphs.emplace_back(graph);
        }
        break; // HACK:Debug only for the first subgraph
    }
    dbg(out_graphs);
    for (auto graph : out_graphs) {
        graph->print();
    }
    cntStates += derivator.getNumIntermediateStates();
    cntCandidates += derivator.getNumCandidates();
}

void NMutator::runMultipleOps(SubGraph *in_graph,
                              std::vector<SubGraph *> &out_graphs) {
    std::cout << "run multiple ops" << std::endl;
    in_graph->print();
    std::cout << std::endl;

    std::vector<Operator *> computeOps;
    dbg(computeOps);
    in_graph->getComputeOps(computeOps);
    nnet::VecExpr exprs;
    for (const auto &op : computeOps)
        exprs.emplace_back(opToExpression(op));
    dbg(exprs);

    nnet::Derivator derivator;
    nnet::MultiFormulas origin(exprs, 0);
    bool canCombine = derivator.stageCombination(origin, 0);
    dbg(canCombine);
    const auto matmul0 = dynamic_cast<MatmulOp *>(computeOps[0]);
    assert(matmul0);
    // Build merged graph
    auto g = new tpm::Graph();
    std::vector<Tensor *> inputsT, weightsT, outputsT;
    for (const auto &opT : computeOps) {
        inputsT.emplace_back(opT->getInputs(0));
        weightsT.emplace_back(opT->getInputs(1));
        outputsT.emplace_back(opT->getOutput());
    }
    const auto concat1 = g->concat(inputsT, 0);
    const auto concat2 = g->concat(weightsT, 0);
    const auto matmul = g->matmul(concat1->getOutput(), concat2->getOutput(),
                                  matmul0->getTransA(), matmul0->getTransB());
    g->split(matmul->getOutput(), outputsT, 0, computeOps.size());
    // Build computation graph in PET:
    g->updateConnection();
    SubGraph *graph = new SubGraph(g->getOperators());
    out_graphs.emplace_back(graph);
    // DEBUG
    dbg(out_graphs);
    for (auto graph : out_graphs) {
        graph->print();
    }
}

NMutator::SGType NMutator::statGraph(SubGraph *sg) {
    auto ops = sg->getOperators();
    switch (ops.size()) {
    case 0: {
        return Empty;
        break;
    }

    case 1: {
        if (ops[0]->getType() == Operator::Conv) {
            auto weight = ops[0]->getInputs()[1];
            auto r = weight->getDims()[2];
            auto s = weight->getDims()[3];
            if (((ConvOp *)sg->getOperators()[0])->getDh() == 1 &&
                ((ConvOp *)sg->getOperators()[0])->getDw() == 1 && r == 1 &&
                s == 1) {
                return Conv1X1;
            } else if (((ConvOp *)sg->getOperators()[0])->getDh() == 2 ||
                       ((ConvOp *)sg->getOperators()[0])->getDw() == 2) {
                return DilatedConv;
            } else {
                const Dim &inDim = ops[0]->getInputs()[0]->getDims();
                const Dim &wDim = ops[0]->getInputs()[1]->getDims();
                if (inDim[2] % 2 == 1 && inDim[3] % 2 == 1)
                    return NormalOddConv;
                else if (wDim[2] != wDim[3])
                    return TransKernelConv;
                else
                    return NormalConv;
            }
        } else if (ops[0]->getType() == Operator::Matmul) {
            return NormalMatmul;
        }
        break;
    }

    default:
        auto ty = ops[0]->getType();
        for (size_t i = 1, iEnd = ops.size(); i < iEnd; ++i) {
            if (ops[i]->getType() != ty)
                return Others;
        }
        if (ty == Operator::Conv) {
            std::vector<ConvOp *> convs;
            for (auto op : ops)
                convs.emplace_back(dynamic_cast<ConvOp *>(op));
            // TODO: 1x1 conv enlarge. 1x1 conv has 0 padding
            for (size_t i = 1, iEnd = ops.size(); i < iEnd; ++i)
                if (!convs[i]->same(*convs[0]))
                    return Others;
            auto inDim = ops[0]->getInputs(0)->getDims();
            // TODO: enlarge input tensor?
            for (size_t i = 1, iEnd = ops.size(); i < iEnd; ++i)
                if (ops[i]->getInputs(0)->getDims() != inDim)
                    return Others;
            auto weightDim = ops[0]->getInputs(1)->getDims();
            auto groupFlag = true;
            // TODO: kernel enlarge to group?
            for (size_t i = 1, iEnd = ops.size(); i < iEnd; ++i) {
                auto wDim = ops[i]->getInputs(1)->getDims();
                if (!(wDim[1] == weightDim[1] && wDim[2] == weightDim[2] &&
                      wDim[3] == weightDim[3] && wDim[2] == wDim[3])) {
                    groupFlag = false;
                    break;
                }
            }
            if (groupFlag)
                return GroupConv;
            // Hack for HetConv
            if (ops.size() == 2) {
                auto w1Dim = weightDim, w2Dim = ops[1]->getInputs(1)->getDims();
                auto hConvFlag = false;
                if (w1Dim[0] == w2Dim[0] && w1Dim[1] == w2Dim[1]) {
                    if (w1Dim[2] == 3 && w1Dim[3] == 3 && w2Dim[2] == 1 &&
                        w2Dim[3] == 1) {
                        hConvFlag = true;
                    }
                    if (w1Dim[2] == 1 && w1Dim[3] == 1 && w2Dim[2] == 3 &&
                        w2Dim[3] == 3) {
                        hConvFlag = true;
                    }
                }
                if (hConvFlag) {
                    // std::cout << "[nmutator stat graph]Het Conv found!"
                    //           << std::endl;
                    // ops[0]->print();
                    // std::cout << std::endl;
                    // ops[1]->print();
                    // std::cout << std::endl;
                    return HetConv;
                }
            }
            auto transGroupFlag = true;
            // TODO: transpose group conv with different f dim?
            for (size_t i = 1, iEnd = ops.size(); i < iEnd; ++i) {
                auto wDim = ops[i]->getInputs(1)->getDims();
                if (!(wDim[0] == weightDim[0] && wDim[1] == weightDim[1] &&
                      ((wDim[2] == weightDim[2] && wDim[3] == weightDim[3]) ||
                       (wDim[2] == weightDim[3] && wDim[3] == weightDim[2])))) {
                    transGroupFlag = false;
                    break;
                }
            }
            if (transGroupFlag)
                return TransposeGroupConv;
        } else if (ty == Operator::Matmul) {
            // check same input shape or not
            for (int i = 0; i < (int)ops.size() - 1; ++i) {
                assert(dynamic_cast<MatmulOp *>(ops[i])->getTransA() ==
                       dynamic_cast<MatmulOp *>(ops[i + 1])->getTransA());
                assert(dynamic_cast<MatmulOp *>(ops[i])->getTransB() ==
                       dynamic_cast<MatmulOp *>(ops[i + 1])->getTransB());
                if (ops[i]->getInputs()[0]->getDims() !=
                    ops[i + 1]->getInputs()[0]->getDims()) {
                    return Others;
                }
                if (ops[i]->getInputs()[1]->getDims() !=
                    ops[i + 1]->getInputs()[1]->getDims()) {
                    return Others;
                }
            }
            return BatchMatmul;
        }
        // TODO: others?
        break;
    }

    return Others;
}

uint64_t NMutator::computeHashForSingleComputeOp(const Operator *op) {
    if (op->getType() == Operator::Conv) {
        auto conv = dynamic_cast<const ConvOp *>(op);
        auto hash = conv->getHash();
        auto inputDim = conv->getInputs()[0]->getDims();
        auto weightDim = conv->getOutputs()[0]->getDims();
        hash += inputDim[0] * 10000019 + inputDim[1] * 10000079 +
                inputDim[2] * 10000103 + inputDim[3] * 10000121 +
                weightDim[0] * 10000139 + weightDim[1] * 10000141 +
                weightDim[2] * 10000169 + weightDim[3] * 10000189;
        return hash;
    } else if (op->getType() == Operator::ConvTrans) {
        auto conv = dynamic_cast<const ConvTransOp *>(op);
        auto hash = conv->getHash();
        auto inputDim = conv->getInputs()[0]->getDims();
        auto weightDim = conv->getOutputs()[0]->getDims();
        hash += inputDim[0] * 10000019 + inputDim[1] * 10000079 +
                inputDim[2] * 10000103 + inputDim[3] * 10000121 +
                weightDim[0] * 10000139 + weightDim[1] * 10000141 +
                weightDim[2] * 10000169 + weightDim[3] * 10000189;
        return hash;
    } else if (op->getType() == Operator::Matmul) {
        static uint64_t matmulhash = 0;
        return matmulhash++;
    } else if (op->getType() == Operator::G2BMM) {
        auto g2bmm = dynamic_cast<const G2BMMOp *>(op);
        auto hash = g2bmm->getHash();
        auto inputDim = g2bmm->getInputs()[0]->getDims();
        auto weightDim = g2bmm->getOutputs()[0]->getDims();
        hash += inputDim[0] * 10000019 + inputDim[1] * 10000079 +
                inputDim[2] * 10000103 + inputDim[3] * 10000121 +
                weightDim[0] * 10000139 + weightDim[1] * 10000141 +
                weightDim[2] * 10000169 + weightDim[3] * 10000189;
        return hash;
    } else if (op->getType() == Operator::GBMML) {
        auto gbmml = dynamic_cast<const GBMMLOp *>(op);
        auto hash = gbmml->getHash();
        auto inputDim = gbmml->getInputs()[0]->getDims();
        auto weightDim = gbmml->getOutputs()[0]->getDims();
        hash += inputDim[0] * 10000019 + inputDim[1] * 10000079 +
                inputDim[2] * 10000103 + inputDim[3] * 10000121 +
                weightDim[0] * 10000139 + weightDim[1] * 10000141 +
                weightDim[2] * 10000169 + weightDim[3] * 10000189;
        return hash;
    } else {
        // Not impl
        assert(false);
        return 0;
    }
}

nnet::Expr NMutator::opToExpression(Operator *op) {
    if (auto convOp = dynamic_cast<ConvOp *>(op)) {
        const auto &inputs = convOp->getInputs();
        const auto &AT = inputs[0];
        const auto &KT = inputs[1];
        const auto &[n, c, h, w, f, r, s, ph, pw, sh, sw, dh, dw, g, bi, ac] =
            convOp->getArgs(0);
        dbg(n, c, h, w, f, r, s, ph, pw, sh, sw, dh, dw);
        if (!(sh == 1 && sw == 1 && dh == 1 && dw == 1))
            return nullptr;
        assert(sh == 1 && sw == 1 && dh == 1 && dw == 1);
        inputsNameNToTensorT["A"] = AT;
        inputsNameNToTensorT["K"] = KT;
        const auto A = nnet::makeTensor("A", AT->getDims(),
                                        std::vector<int>{0, 0, ph, pw});
        const auto K = nnet::makeTensor("K", KT->getDims());
        return nnet::ConvPattern::getExpr(A, K, n, c, h, w, f, r, s);
    } else if (auto convOp = dynamic_cast<ConvTransOp *>(op)) {
        const auto &AT = convOp->getInputs()[0];
        const auto &KT = convOp->getInputs()[1];
        inputsNameNToTensorT["A"] = AT;
        inputsNameNToTensorT["K"] = KT;
        const auto &[n, c, h, w, f, r, s, ph, pw, sh, sw, dh, dw, g, bi, ac] =
            convOp->getArgs(0);
        if (r != 4) {
            dbg("ConvTranspose R!=4. Skipped.", r);
            return nullptr;
        }
        int padding = 1 * (r - 1) - 1;
        const auto A = nnet::makeTensor(
            "A", AT->getDims(), std::vector<int>{0, padding, padding, 0});
        const auto K = nnet::makeTensor("K", KT->getDims());
        return nnet::ConvTransPattern::getExpr(A, K, n, c, h, w, f, r, s);
    } else if (auto g2bmmOp = dynamic_cast<G2BMMOp *>(op)) {
        const auto &AT = g2bmmOp->getInputs()[0];
        const auto &BT = g2bmmOp->getInputs()[1];
        const auto [b, m, k, width, dilation] = g2bmmOp->getArgs();

        const auto &[expr, inputsN] =
            nnet::Sg2bmmPattern::getExpr(b, m, k, width, dilation);
        inputsNameNToTensorT[inputsN.first->getName()] = AT;
        inputsNameNToTensorT[inputsN.second->getName()] = BT;
        return expr;
    } else if (auto gbmmlOp = dynamic_cast<GBMMLOp *>(op)) {
        const auto &AT = gbmmlOp->getInputs()[0];
        const auto &BT = gbmmlOp->getInputs()[1];
        const auto [b, m, w, k, dilation] = gbmmlOp->getArgs();
        const auto &[expr, inputsN] =
            nnet::LongformerGBMMPattern::getExpr(b, m, w, k, dilation);
        inputsNameNToTensorT[inputsN.first->getName()] = AT;
        inputsNameNToTensorT[inputsN.second->getName()] = BT;
        dbg(b, m, w, k, dilation, expr);
        return expr;
    } else if (auto matmulOp = dynamic_cast<MatmulOp *>(op)) {
        const auto &AT = matmulOp->getInputs()[0];
        const auto &BT = matmulOp->getInputs()[1];
        const auto [transA, transB, b, m, n, k] = matmulOp->getArgs();
        const auto &[expr, inputsN] =
            nnet::MatmulPattern::getExpr(transA, transB, b, m, n, k);
        inputsNameNToTensorT[inputsN.first->getName()] = AT;
        inputsNameNToTensorT[inputsN.second->getName()] = BT;
        dbg(b, m, n, k, expr);
        return expr;
    }
    // else if (auto transposeOp = dynamic_cast<TransposeOp *>(op)) {
    //     return transposeOpToExpression(transposeOp);
    // }
    nnet_unimplemented_continue();
    return nullptr;
}

tpm::SubGraph *NMutator::fuseHetConv(nnet::Expr expr, SubGraph *in_graph) {
    // Conv3x3+Conv1x1 => Gemm(nhw, f(rs+1), c) + Reduce
    auto g = std::make_shared<tpm::Graph>();
    in_graph->print();
    assert(in_graph->getInputs().size() == 3);
    auto input = in_graph->getOperators()[0]->getInputs(0);
    auto conv = dynamic_cast<ConvOp *>(in_graph->getOperators()[0]);
    auto output = conv->getOutput();
    // auto input = g->reshape(input);
    auto inputTrans = g->transpose(input, 0, {-1, {0, 2, 3}, 1}, -1);
    // dbg(inputTrans->getOutput()->getDims());
    const auto &[n, c, h, w, f, r, s, ph, pw, sh, sw, dh, dw, G, bi, ac] =
        conv->getArgs(0);
    auto weight = g->tensor({1, c, f * (3 * 3 + 1)});
    dbg(weight->getDims());
    auto matmul = g->matmul(inputTrans->getOutput(), weight, false, false);
    auto bias = g->tensor({f});
    const double size = n * f * h * w * (3 * 3 + 1) * 4;
    // FIXME: add NNET tensors for verfication
    auto membound =
        g->membound({matmul->getOutput(), bias}, {output}, {}, nullptr,
                    memboundTime(size), "Reduce_conv3x3+1x1");
    dbg(n, f, h, w);
    dynamic_cast<MemBoundOp *>(membound)->setNFHW(n, f, h, w);

    return new SubGraph(g->getOperators());
}

tpm::SubGraph *NMutator::expressionToGraph(nnet::Expr expr,
                                           SubGraph *in_graph) {
    auto g = new tpm::Graph();
    nnet::FullPrinterVisitor fullVisitor;
    const auto &tensorQueueN = fullVisitor.traverse(expr);
    // Build tensors: Skip the first one, which is output
    auto nameNToTensorT = inputsNameNToTensorT;
    for (size_t i = 1; i < tensorQueueN.size(); ++i) {
        const auto &[nameN, routineN, tensorN] = tensorQueueN[i];
        // dbg(nameN, routineN, tensorN);
        if (!routineN) {
            // This is an inputs
            assert(nameNToTensorT.count(nameN));
        } else {
            assert(!nameNToTensorT.count(nameN));
            nameNToTensorT[nameN] = g->tensor(tensorN->getShape());
        }
    }
    const auto &outputsPET = in_graph->getOutputs();
    if (outputsPET.size() != 1) {
        nnet_unimplemented_continue();
        return nullptr;
    }
    nameNToTensorT[std::get<0>(tensorQueueN.at(0))] = outputsPET[0];
    // Build computation graph in PET:
    for (int i = tensorQueueN.size() - 1; i >= 0; --i) {
        const auto &[outputNameN, routineN, tensorN] = tensorQueueN[i];
        if (!routineN)
            continue;
        // dbg(outputNameN, routineN, tensorN, routineN->getType());
        if (auto op = nnet::as<nnet::ConvNode>(routineN)) {
            // g->conv(i8, w9, 2, 2);
            std::vector<nnet::Tensor> inputsN = op->getInputs();
            auto A = nameNToTensorT.at(inputsN[0]->getName());
            auto K = nameNToTensorT.at(inputsN[1]->getName());
            auto output = nameNToTensorT.at(outputNameN);
            const auto &[ph, pw, sh, sw, dh, dw] = op->getArgs();
            g->conv(A, K, output, ph, pw, sh, sw, dh, dw);
        } else if (auto op = nnet::as<nnet::ElementWiseNode>(routineN)) {
            assert(op->getInputs().size() == 1);
            TensorVec inputsPET;
            TensorVec outputsPET = {nameNToTensorT.at(outputNameN)};
            for (const auto &inputN : op->getInputs())
                inputsPET.emplace_back(nameNToTensorT.at(inputN->getName()));
            // Re-estimate time here.
            ssize_t cnt = 0;
            for (const auto tensor : inputsPET)
                cnt += tensor->size();
            for (const auto tensor : outputsPET)
                cnt += tensor->size();
            g->membound(inputsPET, outputsPET, op->getInputs(), op->getExpr(),
                        memboundTime(cnt));
        } else if (auto op = nnet::as<nnet::MatmulNode>(routineN)) {
            assert(op->getInputs().size() == 2);
            nnet::Tensor AN = op->getInputs()[0];
            nnet::Tensor BN = op->getInputs()[1];
            TensorVec inputsPET = {nameNToTensorT.at(AN->getName()),
                                   nameNToTensorT.at(BN->getName())};
            TensorVec outputsPET = {nameNToTensorT.at(outputNameN)};
            const auto &[b, m, n, k, transa, transb] = op->getArgs();
            g->matmul(inputsPET[0], inputsPET[1], outputsPET[0], transa,
                      transb);
        } else if (auto op = nnet::as<nnet::G2bmmNode>(routineN)) {
            assert(op->getInputs().size() == 2);
            nnet::Tensor AN = op->getInputs()[0];
            nnet::Tensor BN = op->getInputs()[1];
            TensorVec inputsPET = {nameNToTensorT.at(AN->getName()),
                                   nameNToTensorT.at(BN->getName())};
            TensorVec outputsPET = {nameNToTensorT.at(outputNameN)};
            const auto &[b, m, w, k, dilation] = op->getArgs();
            g->g2bmm(inputsPET[0], inputsPET[1], outputsPET[0], w, dilation);
        } else if (auto op = nnet::as<nnet::GbmmNode>(routineN)) {
            assert(op->getInputs().size() == 2);
            nnet::Tensor AN = op->getInputs()[0];
            nnet::Tensor BN = op->getInputs()[1];
            TensorVec inputsPET = {nameNToTensorT.at(AN->getName()),
                                   nameNToTensorT.at(BN->getName())};
            TensorVec outputsPET = {nameNToTensorT.at(outputNameN)};
            const auto &[b, m, w, n, dilation] = op->getArgs();
            g->gbmml(inputsPET[0], inputsPET[1], outputsPET[0], dilation);
        }
    }
    g->updateConnection();
    SubGraph *graph = new SubGraph(g->getOperators());
    return graph;
}

SubGraph *NMutator::transformDialtedConv(Operator *op) {
    if (auto convOp = dynamic_cast<ConvOp *>(op)) {
        if (convOp->getPh() == convOp->getDh() && convOp->getSh() == 1 &&
            convOp->getDh() > 1 && convOp->getDh() == convOp->getDw()) {
            const int d = convOp->getDh();
            assert(convOp->getInputs()[0]->getDims()[2] % d == 0);
            auto g = new tpm::Graph();
            auto inputDims = convOp->getInputs(0)->getDims();
            auto weightDims = convOp->getInputs(1)->getDims();
            auto outputDims = convOp->getOutput()->getDims();
            auto newA = g->tensor({inputDims[0] * d * d, inputDims[1],
                                   inputDims[2] / d, inputDims[3] / d});
            // auto newW = g->tensor(
            //     {weightDims[0] * weightDims[1] * weightDims[3],
            //     weightDims[2]});
            auto newO =
                g->tensor({inputDims[0] * inputDims[1] * inputDims[2],
                           weightDims[0] * weightDims[1] * weightDims[3]});
            g->membound(
                {convOp->getInputs(0)}, {newA}, {}, nullptr,
                memboundTime(convOp->getInputs(0)->size() + newA->size()),
                "DConv Tranpose Input");
            // g->membound({convOp->getInputs(1)}, {newW}, {}, nullptr, 0,
            //             "Tranpose Weight");
            g->conv(newA, convOp->getInputs(1), newO, 1, 1, 1, 1);
            g->reshape(newO, convOp->getOutput());
            dbg(newO->size(), convOp->getOutput()->size());
            assert(newO->size() == convOp->getOutput()->size());
            g->membound(
                {newO}, {convOp->getOutput()}, {}, nullptr,
                memboundTime(newO->size() + convOp->getOutput()->size()),
                "DConv Tranpose Output");
            g->updateConnection();
            SubGraph *graph = new SubGraph(g->getOperators());
            return graph;
        }
    }
    return nullptr;
}

double NMutator::memboundTime(ssize_t cnt) {
    return double(cnt) * 4 / bandwidth * 1000; // millisecond
}

double NMutator::memboundTime(const Dim &dims) {
    return memboundTime(dims.size());
}

SubGraph *NMutator::transformTConv3x3(Operator *op) {
    if (auto tconvOp = dynamic_cast<ConvTransOp *>(op)) {
        dbg(tconvOp->getInputs()[1]->getDims());
        if (tconvOp->getPh() == 1 && tconvOp->getSh() == 2 &&
            tconvOp->getInputs()[1]->getDims()[0] == 3 &&
            tconvOp->getInputs()[1]->getDims()[1] == 3) {
            auto g = new tpm::Graph();
            auto inputDims = tconvOp->getInputs(0)->getDims();
            auto weightDims = tconvOp->getInputs(1)->getDims();
            auto outputDims = tconvOp->getOutput()->getDims();
            // NHWF
            auto newA = g->tensor(
                {inputDims[0] * inputDims[1] * inputDims[2], inputDims[3]});
            // RSFC
            auto newW = g->tensor(
                {weightDims[0] * weightDims[1] * weightDims[3], weightDims[2]});
            auto newO =
                g->tensor({inputDims[0] * inputDims[1] * inputDims[2],
                           weightDims[0] * weightDims[1] * weightDims[3]});
            g->reshape(tconvOp->getInputs(0), newA);
            g->reshape(tconvOp->getInputs(1), newW);
            g->matmul(newA, newW, newO, 0, 1);
            // g->reshape(newO, tconvOp->getOutput());
            tconvOp->print();
            dbg(newO->size() * 4, tconvOp->getOutput()->size() * 9);
            assert(newO->size() * 4 == tconvOp->getOutput()->size() * 9);
            g->membound(
                {newO}, {tconvOp->getOutput()}, {}, nullptr,
                memboundTime(newO->size() + tconvOp->getOutput()->size()),
                "TConv3x3 reduce");
            g->updateConnection();
            SubGraph *graph = new SubGraph(g->getOperators());
            return graph;
        }
    }
    return nullptr;
}

SubGraph *NMutator::transformTConv1x1(Operator *op) {
    if (auto tconvOp = dynamic_cast<ConvTransOp *>(op)) {
        if (tconvOp->getPh() == 0 && tconvOp->getSh() == 1) {
            auto g = new tpm::Graph();
            auto inputDims = tconvOp->getInputs(0)->getDims();
            auto weightDims = tconvOp->getInputs(1)->getDims();
            auto outputDims = tconvOp->getOutput()->getDims();
            auto newA = g->tensor(
                {inputDims[0] * inputDims[1] * inputDims[2], inputDims[3]});
            auto newW = g->tensor(
                {weightDims[0] * weightDims[1] * weightDims[3], weightDims[2]});
            auto newO =
                g->tensor({inputDims[0] * inputDims[1] * inputDims[2],
                           weightDims[0] * weightDims[1] * weightDims[3]});
            g->reshape(tconvOp->getInputs(0), newA);
            g->reshape(tconvOp->getInputs(1), newW);
            g->matmul(newA, newW, newO, 0, 1);
            g->reshape(newO, tconvOp->getOutput());
            g->updateConnection();
            SubGraph *graph = new SubGraph(g->getOperators());
            return graph;
        }
    }
    return nullptr;
}

SubGraph *NMutator::transformConv1x1(Operator *op) {
    auto convOp = dynamic_cast<ConvOp *>(op);
    if (!convOp)
        return nullptr;
    if (convOp->getPh() == 0 && convOp->getSh() == 1 &&
        convOp->getInputs()[1]->getDims()[2] == 1 &&
        convOp->getInputs()[1]->getDims()[3] == 1) {
        // Transpose is requrired for BS>1
        // if (convOp->getInputs()[0]->getDims()[0] == 1) {
        auto g = new tpm::Graph();
        auto inputDims = convOp->getInputs(0)->getDims();
        auto weightDims = convOp->getInputs(1)->getDims();
        auto outputDims = convOp->getOutput()->getDims();
        auto newA = g->tensor(
            {inputDims[1], inputDims[0] * inputDims[2] * inputDims[3]});
        auto newW = g->tensor({weightDims[0], weightDims[1]});
        auto newO = g->tensor(
            {weightDims[0], inputDims[0] * inputDims[2] * inputDims[3]});
        g->reshape(convOp->getInputs(0), newA);
        g->reshape(convOp->getInputs(1), newW);
        g->matmul(newW, newA, newO, 0, 0);
        g->reshape(newO, convOp->getOutput());
        g->updateConnection();
        SubGraph *graph = new SubGraph(g->getOperators());
        return graph;
    }
    return nullptr;
}

SubGraph *NMutator::transformConv1xk(Operator *op) {
    auto convOp = dynamic_cast<ConvOp *>(op);
    if (!convOp)
        return nullptr;
    if (convOp->getSh() != 1 || convOp->getSw() != 1)
        return nullptr;
    bool a = convOp->getInputs()[1]->getDims()[2] == 1;
    bool b = convOp->getInputs()[1]->getDims()[3] == 1;
    if (!(a ^ b))
        return nullptr;
    convOp->print();
    auto g = new tpm::Graph();
    auto inputDims = convOp->getInputs(0)->getDims();
    auto weightDims = convOp->getInputs(1)->getDims();
    auto outputDims = convOp->getOutput()->getDims();
    auto newA =
        g->tensor({inputDims[0] * inputDims[2] * inputDims[3], inputDims[1]});
    auto newW = g->tensor(
        {weightDims[0] * weightDims[2] * weightDims[3], weightDims[1]});
    auto newO = g->tensor({weightDims[0] * weightDims[2] * weightDims[3],
                           inputDims[0] * inputDims[2] * inputDims[3]});
    // g->reshape(convOp->getInputs(0), newA);
    g->membound({convOp->getInputs(0)}, {newA}, {}, nullptr,
                memboundTime(convOp->getInputs(0)->size() + newA->size()),
                "1xk input reshape");
    g->reshape(convOp->getInputs(1), newW);

    g->matmul(newW, newA, newO, 0, 1);
    g->membound({newO}, {convOp->getOutput()}, {}, nullptr,
                memboundTime(newW->size() + convOp->getOutput()->size()),
                "1xk reduce");
    g->updateConnection();
    SubGraph *graph = new SubGraph(g->getOperators());
    return graph;
}

} // namespace tpm
