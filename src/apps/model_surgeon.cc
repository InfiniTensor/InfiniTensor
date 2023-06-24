#include "core/graph.h"
#include "core/runtime.h"
#include "nnet/dbg.h"
#include "operators/conv.h"
#include "operators/pooling.h"
#include "operators/reshape.h"

namespace infini {
Graph convertNCHWtoNHWCModel(Runtime runtime, Graph inG) {
    // Construct new graph
    // IT_ASSERT(inG->getInputs().size() == 1);
    IT_ASSERT(inG->getOutputs().size() == 1);
    bool status = inG->topo_sort();
    IT_ASSERT(status);
    auto g = make_ref<GraphObj>(runtime);
    map<UidBaseType, Tensor> tensors;
    for (const auto &t : inG->getTensors())
        if (t->getDims().size() != 4)
            return nullptr;
    auto getTensor = [&g, &tensors](const Tensor &inTensor) {
        auto uid = inTensor->getGuid();
        if (auto it = tensors.find(uid); it == tensors.end()) {
            Shape s = inTensor->getDims();
            s = vector{s[0], s[2], s[3], s[1]};
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
            g->addOpWithOutputs<ConvNHWCObj>(inputs[0], inputs[1], outputs[0],
                                             ph, pw, sh, sw, dh, dw, bias,
                                             cOp->getAct());
        } else if (const auto &cOp = as<ConvTransposed2dObj>(op)) {
            const auto &[ph, pw, sh, sw, dh, dw] = cOp->getPadStrideDilation();
            const auto &[oph, opw] = cOp->getOutputPadding();
            auto group = cOp->getNumGroups();
            auto bias =
                cOp->getBias() ? g->cloneTensor(cOp->getBias()) : nullptr;
            g->addOpWithOutputs<ConvTransposed2dNHWCObj>(
                inputs[0], inputs[1], outputs[0], ph, pw, sh, sw, dh, dw, oph,
                opw, group, bias, cOp->getAct());
        } else if (const auto &cOp = as<MaxPoolObj>(op)) {
            auto t = g->addOp<ReshapeObj>(inputs[0], nullptr,
                                          cOp->getInputs(0)->getDims())
                         ->getOutput();
            auto tt = g->addTensor(cOp->getOutput()->getDims(),
                                   cOp->getOutput()->getDType());
            g->cloneOperator(op, {t}, {tt});
            g->addOpWithOutputs<ReshapeObj>(tt, outputs[0]);
        } else {
            dbg(op);
            g->cloneOperator(op, inputs, outputs);
        }
    }
    return g;
}
} // namespace infini