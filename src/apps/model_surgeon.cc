#include "core/graph.h"
#include "core/runtime.h"
#include "nnet/dbg.h"
#include "operators/concat.h"
#include "operators/conv.h"
#include "operators/matmul.h"
#include "operators/pooling.h"
#include "operators/reshape.h"
#include "operators/transpose.h"

#ifdef BANG
#include "bang/bang_runtime.h"
#endif // BANG
namespace infini {

Tensor runWeightComputation(const Tensor &weight) {
#ifdef BANG
    auto rt = make_ref<NativeCpuRuntimeObj>();
    auto g = make_ref<GraphObj>(rt);
    auto in = g->addTensor(weight);
    auto out = g->addOp<TransposeObj>(weight, nullptr, vector<int>{0, 2, 3, 1})
                   ->getOutput();
    g->dataMalloc();
    g->getRuntime()->run(g);
    return g->getOutputs()[0];
#else
    IT_TODO_HALT();
    return nullptr;
#endif // BANG
}

Graph convertNCHWtoNHWCModel(Graph inG) {
    // Construct new graph
    // IT_ASSERT(inG->getInputs().size() == 1);
    {
        auto useless = vector<Tensor>();
        for (const auto &t : inG->getTensors()) {
            if (!t->getSource() && t->getTargets().empty()) {
                useless.push_back(t);
            }
        }
        for (auto t : useless) {
            inG->removeTensor(t);
        }
    }
    IT_ASSERT(inG->getOutputs().size() == 1);
    bool status = inG->topo_sort();
    IT_ASSERT(status);
    auto g = make_ref<GraphObj>(inG->getRuntime());
    map<UidBaseType, Tensor> tensors;
    // modelStatus: if currently processing Conv-related subgraph
    // 0: before processcing Conv-related subgraph
    // 1: processing Conv-related subgraph
    // 2: after processing Conv-related subgraph
    int modelStatus = 0;
    for (const auto &t : inG->getTensors())
        if (t->getDims().size() != 4)
            return nullptr;
    auto getTensor = [&g, &tensors](const Tensor &inTensor) {
        auto uid = inTensor->getGuid();
        if (auto it = tensors.find(uid); it == tensors.end()) {
            Shape s = inTensor->getDims();
            // Only transpose 4-dimension tensors
            if (s.size() == 4) {
                s = vector{s[0], s[2], s[3], s[1]};
            }
            tensors[uid] = g->addTensor(s, inTensor->getDType(),
                                        inTensor->getTensorType());
        }
        return tensors[uid];
    };
    for (auto op : inG->getOperators()) {
        TensorVec inputs, outputs;
        for (auto &t : op->getInputs())
            inputs.emplace_back(getTensor(t));
        for (auto &t : op->getOutputs())
            outputs.emplace_back(getTensor(t));
        if (modelStatus == 1) {
            if (auto cOp = as<ConvObj>(op)) {
                const auto &[ph, pw, sh, sw, dh, dw] =
                    cOp->getPadStrideDilation();
                auto bias =
                    cOp->getBias() ? g->cloneTensor(cOp->getBias()) : nullptr;
                auto weight = runWeightComputation(inputs[1]);
                g->addTensor(weight);
                g->addOpWithOutputs<ConvNHWCObj>(inputs[0], weight, outputs[0],
                                                 ph, pw, sh, sw, dh, dw, bias,
                                                 cOp->getAct());
            } else if (const auto &cOp = as<ConvTransposed2dObj>(op)) {
                const auto &[ph, pw, sh, sw, dh, dw] =
                    cOp->getPadStrideDilation();
                const auto &[oph, opw] = cOp->getOutputPadding();
                auto group = cOp->getNumGroups();
                auto bias =
                    cOp->getBias() ? g->cloneTensor(cOp->getBias()) : nullptr;
                auto weight = runWeightComputation(inputs[1]);
                g->addTensor(weight);
                g->addOpWithOutputs<ConvTransposed2dNHWCObj>(
                    inputs[0], weight, outputs[0], ph, pw, sh, sw, dh, dw, oph,
                    opw, group, bias, cOp->getAct());
            } else if (const auto &pOp = as<PoolingObj>(op)) {
                auto t = g->addOp<TransposeObj>(inputs[0], nullptr,
                                                vector<int>{0, 2, 3, 1})
                             ->getOutput();
                auto tt = g->addTensor(op->getOutput()->getDims(),
                                       op->getOutput()->getDType());
                g->cloneOperator(op, {t}, {tt});
                g->addOpWithOutputs<TransposeObj>(tt, outputs[0],
                                                  vector<int>{0, 3, 1, 2});
            } else if (const auto &ccOp = as<ConcatObj>(op)) {
                int axis = ccOp->getDim();
                axis = vector<int>{0, 3, 1, 2}[axis];
                g->addOpWithOutputs<ConcatObj>(inputs, outputs[0], axis);
            } else if (const auto &fOp = as<FlattenObj>(op)) {
                IT_ASSERT(inputs[0]->getDims().size() == 4 &&
                          outputs[0]->getDims().size() == 2);
                IT_ASSERT(fOp->getAxis() == 1);
                g->cloneOperator(fOp, inputs, outputs);
                modelStatus = 2;
            } else {
                dbg(op);
                // If this operator is not specially handled, and all of its
                // intputs and outputs are 4-dimension, insert transpose
                // operators before and after this operator.
                // The dimensions of all inputs and outputs must be 4 when
                // modelStatus is 1.
                for (auto &t : inputs) {
                    if (t->getDims().size() != 4)
                        IT_TODO_HALT();
                }
                for (auto &t : outputs) {
                    if (t->getDims().size() != 4)
                        IT_TODO_HALT();
                }
                auto t = g->addOp<TransposeObj>(inputs[0], nullptr,
                                                vector<int>{0, 2, 3, 1})
                             ->getOutput();
                auto tt = g->addTensor(op->getOutput()->getDims(),
                                       op->getOutput()->getDType());
                g->cloneOperator(op, {t}, {tt});
                g->addOpWithOutputs<TransposeObj>(tt, outputs[0],
                                                  vector<int>{0, 3, 1, 2});
            }
        } else {
            if (auto cOp = as<ConvObj>(op)) {
                if (modelStatus == 0) {
                    const auto &[ph, pw, sh, sw, dh, dw] =
                        cOp->getPadStrideDilation();
                    auto bias = cOp->getBias() ? g->cloneTensor(cOp->getBias())
                                               : nullptr;
                    auto t = g->addOp<TransposeObj>(inputs[0], nullptr,
                                                    vector<int>{0, 2, 3, 1})
                                 ->getOutput();
                    auto weight = runWeightComputation(inputs[1]);
                    g->addTensor(weight);
                    g->addOpWithOutputs<ConvNHWCObj>(t, weight, outputs[0], ph,
                                                     pw, sh, sw, dh, dw, bias,
                                                     cOp->getAct());
                    modelStatus = 1;
                } else {
                    IT_TODO_HALT();
                }
            } else if (const auto &cOp = as<ConvTransposed2dObj>(op)) {
                IT_TODO_HALT();
            } else {
                g->cloneOperator(op, inputs, outputs);
            }
        }
    }
    return g;
}
} // namespace infini
