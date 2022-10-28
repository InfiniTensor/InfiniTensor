#include "pfusion/meta_graph.h"

namespace memb {

void MetaGraph::print() {
    for (auto op : metaOps) {
        op->print();
    }
}

void MetaGraph::optimize() {
    std::vector<std::shared_ptr<MetaOp>> ops;
    int numOp = metaOps.size();
    int cur = 0;
    for (int i = 1; i < numOp; i++) {
        auto next = MetaOp::merge(metaOps[cur], metaOps[i]);
        if (next == nullptr) {
            ops.emplace_back(metaOps[cur]);
            cur = i;
        } else {
            cur = metaOps.size();
            metaOps.emplace_back(next);
        }
    }
    ops.emplace_back(metaOps[cur]);
    metaOps.clear();
    for (auto op : ops) {
        op->optimize();
        metaOps.emplace_back(op);
    }
}

std::string MetaGraph::genHeader() {
    std::string code = "#include \"cuda_utils.h\"\n";
    return code;
}

std::string MetaGraph::genKernelFuncs() {
    std::string code = "";
    for (auto metaOp : metaOps) {
        code += metaOp->genKernelFunc();
    }
    return code;
}

std::string MetaGraph::genInvokeFuncs() {
    std::string code = "";
    for (auto metaOp : metaOps) {
        code += metaOp->genInvokeFunc();
    }
    return code;
}

} // namespace memb