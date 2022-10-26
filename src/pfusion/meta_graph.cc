#include "pfusion/meta_graph.h"

namespace memb {

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