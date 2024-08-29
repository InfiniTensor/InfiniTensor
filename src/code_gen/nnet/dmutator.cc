#include "code_gen/nnet/dmutator.h"

using namespace tpm;

DMutator::DMutator() {}

DMutator::~DMutator() {}

void DMutator::run(SubGraph *in_graph, std::vector<SubGraph *> &out_graphs,
                   int mdepth,
                   std::vector<std::shared_ptr<Operator>> candidate_ops,
                   float threshold) {
    out_graphs.emplace_back(in_graph);
    return; // DEBUG
}

DMutator::SGType DMutator::statGraph(SubGraph *sg) {
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

uint64_t DMutator::computeHashForSingleComputeOp(const Operator *op) {
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
    } else {
        assert(false);
        return 0;
    }
}