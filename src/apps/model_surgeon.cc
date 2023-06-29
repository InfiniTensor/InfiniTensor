#include "core/graph.h"
#include "core/runtime.h"
#include "nnet/dbg.h"
#include "operators/batch_norm.h"
#include "operators/concat.h"
#include "operators/conv.h"
#include "operators/element_wise.h"
#include "operators/gather.h"
#include "operators/matmul.h"
#include "operators/pooling.h"
#include "operators/reshape.h"
#include "operators/transpose.h"
#include "operators/unary.h"
#include "operators/batch_norm.h"
#include "operators/reduce_mean.h"

#ifdef USE_BANG
#include "bang/bang_runtime.h"
#endif // BANG
namespace infini {

Tensor convertData(Runtime &rt, const Tensor &weight, vector<int> permute) {
    IT_ASSERT(weight->getDims().size() == permute.size());
#ifdef USE_BANG
    auto g = make_ref<GraphObj>(rt);
    auto in = g->addTensor(weight);
    auto out = g->addOp<TransposeObj>(weight, nullptr, permute)->getOutput();
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
        auto rt = g->getRuntime();
        auto uid = inTensor->getGuid();
        if (auto it = tensors.find(uid); it == tensors.end()) {
            auto targets = inTensor->getTargets();
            if (std::any_of(targets.begin(), targets.end(), [](const auto &op) {
                    return as<GatherObj>(op) != nullptr;
                })) {

                tensors[uid] = g->addTensor(inTensor);

            } else if (inTensor->getDims().size() != 4) {

                tensors[uid] = g->addTensor(inTensor);

            } else if (inTensor->hasData()) {
                tensors[uid] =
                    g->addTensor(convertData(rt, inTensor, {0, 2, 3, 1}));
            } else {
                Shape s = inTensor->getDims();
                tensors[uid] = g->addTensor(vector{s[0], s[2], s[3], s[1]},
                                            inTensor->getDType(),
                                            inTensor->getTensorType());
            }
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
        } else if (const auto &uOp = as<UnaryObj>(op)) {
            g->cloneOperator(uOp, inputs, outputs);
        } else if (const auto &eOp = as<ElementWiseObj>(op)) {
            g->cloneOperator(eOp, inputs, outputs);
        } else if (const auto &eOp = as<BatchNormObj>(op)) {
            float momentum = eOp->getMomentum();
            float eps = eOp->getEps();
            bool mode = eOp->getTrainingMode();
            g->addOpWithOutputs<BatchNormNHWCObj>(inputs[0], outputs[0], inputs[1], inputs[2], inputs[3], inputs[4], momentum, eps, mode);
        } else if (const auto &eOp = as<ReduceMeanObj>(op)) {
            std::vector<int> axes_vector;
            auto axes = eOp->getAxes(); 
            std::set<int>::iterator it;

            auto convertxxx = [&](std::vector<int> permute){
                for (auto x:axes){
                  axes_vector.push_back(permute[x]);
                }
            };

            switch(inputs[0]->getDims().size()){
              case 4: convertxxx({0,3,1,2});
                      break;
              case 3: convertxxx({0,2,1});
                      break;
              case 2: convertxxx({0,1});
                      break;
              case 1: convertxxx({0});
                      break;
            }
            bool keepDims = eOp->getKeepDims();
            g->addOp<ReduceMeanObj>(inputs[0], nullptr, axes_vector, keepDims);
        } else {
            dbg(op);
            std::cout << OpRegistry::getOpName(op->getOpType()) << std::endl;
            for (auto &t : inputs) {
                if (t->getDims().size() != 4)
                    IT_TODO_HALT();
            }
            for (auto &t : outputs) {
                std::cout<< t->getDims().size() << std::endl;
                if (t->getDims().size() != 4)
                    IT_TODO_HALT();
            }
            // FIXME: the weights for these operators should not be processed
            auto t = g->addOp<TransposeObj>(inputs[0], nullptr,
                                            vector<int>{0, 3, 1, 2})
                         ->getOutput();
            t->dataMalloc();
            auto s = op->getOutput()->getDims();
            auto tt = g->addTensor(s, op->getOutput()->getDType());
            tt->dataMalloc();
            g->cloneOperator(op, {t}, {tt});
            g->addOpWithOutputs<TransposeObj>(tt, outputs[0],
                                              vector<int>{0, 2, 3, 1});
        }
    }
    return g;
}
} // namespace infini
