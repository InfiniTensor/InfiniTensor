#include "core/graph.h"
#include "core/runtime.h"
#include "nnet/dbg.h"
#include "operators/concat.h"
#include "operators/conv.h"
#include "operators/matmul.h"
#include "operators/pooling.h"
#include "operators/reshape.h"
#include "operators/transpose.h"

#ifdef USE_BANG
#include "bang/bang_runtime.h"
#endif // BANG
namespace infini {

Tensor runWeightComputation(Runtime &rt, const Tensor &weight) {
#ifdef USE_BANG
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
    auto runtime = inG->getRuntime();
    auto g = make_ref<GraphObj>(runtime);
    map<UidBaseType, Tensor> tensors;
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
        if (auto cOp = as<ConvObj>(op)) {
            const auto &[ph, pw, sh, sw, dh, dw] = cOp->getPadStrideDilation();
            auto bias =
                cOp->getBias() ? g->cloneTensor(cOp->getBias()) : nullptr;
            auto weight = runWeightComputation(runtime, inputs[1]);
            g->addTensor(weight);
            g->addOpWithOutputs<ConvNHWCObj>(inputs[0], weight, outputs[0], ph,
                                             pw, sh, sw, dh, dw, bias,
                                             cOp->getAct());
        } else if (const auto &cOp = as<ConvTransposed2dObj>(op)) {
            const auto &[ph, pw, sh, sw, dh, dw] = cOp->getPadStrideDilation();
            const auto &[oph, opw] = cOp->getOutputPadding();
            auto group = cOp->getNumGroups();
            auto bias =
                cOp->getBias() ? g->cloneTensor(cOp->getBias()) : nullptr;
            auto weight = runWeightComputation(runtime, inputs[1]);
            g->addTensor(weight);
            g->addOpWithOutputs<ConvTransposed2dNHWCObj>(
                inputs[0], weight, outputs[0], ph, pw, sh, sw, dh, dw, oph, opw,
                group, bias, cOp->getAct());
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
        } else if (const auto &rOp = as<ReshapeObj>(op)) {
            g->addOpWithOutputs<ReshapeObj>(inputs[0], outputs[0],
                                            outputs[0]->getDims());
        } else if (const auto &mmOp = as<MatmulObj>(op)) {
            g->cloneOperator(mmOp, inputs, outputs);
        } else {
            dbg(op);
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
    }
    return g;
}
} // namespace infini
