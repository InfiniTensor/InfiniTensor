#include "pfusion/meta_op.h"

namespace memb {

std::string genOffset(std::string name,
                      std::shared_ptr<TensorMapping> mapping) {
    if (mapping == nullptr) {
        return "";
    }
    std::string code = "int " + name + " = 0;\n";
    std::vector<int> stride;
    auto &map = mapping->map;
    auto &shape = mapping->shape;
    stride.emplace_back(1);
    for (size_t i = 1; i < shape.size(); i++) {
        stride.emplace_back(stride[i - 1] * shape[i - 1]);
    }
    std::string tmpName = "tmp_" + name;
    code += "int " + tmpName + " = loop_idx;\n";
    for (size_t i = 0; i < map.size(); i++) {
        code += name + " += " + tmpName + " % " +
                std::to_string(shape[map[i]]) + " * " +
                std::to_string(stride[map[i]]) + ";\n";
        code += tmpName + " /= " + std::to_string(shape[map[i]]) + ";\n";
    }

    return code;
}

std::string MetaGraph::genHeader() {
    std::string code = "#include \"cuda_utils.h\"\n";
    return code;
}

std::string MetaGraph::genKernelFunc() {
    std::vector<std::shared_ptr<MetaOp>> metaOps;
    for (auto &node : nodes) {
        metaOps.emplace_back(node.metaOps[0]);
    }
    std::string code = "";
    for (auto metaOp : metaOps) {
        code += metaOp->genKernelFunc();
    }
    return code;
}

std::string MetaOp::genKernelFunc() {
    std::string code = "";
    code += "// Kernel\n";
    code += "__global__ void kernel_func_" + std::to_string(id) + "(";
    IT_ASSERT(ptrs.size() > 0);
    code += "float *" + ptrs[0]->getName();
    for (size_t i = 1; i < ptrs.size(); i++) {
        code += ", float *" + ptrs[i]->getName();
    }
    code += ") {\n";
    code += "int lane_id = threadIdx.x % 32;\n";
    code += "int warp_id = threadIdx.x / 32;\n";
    code += "int parallel_idx = blockIdx.x * " + std::to_string(numWarps) +
            " + warp_id;\n";
    if (numReg != 0) {
        code += "float buf[" + std::to_string(numReg) + "];\n";
    }
    if (numSmem != 0) {
        code += "__shared__ float smem[" + std::to_string(numSmem * numWarps) +
                "];\n";
    }

    code += "for (int loop_idx = parallel_idx; loop_idx < " +
            std::to_string(main_loop_ed) +
            "; loop_idx += " + std::to_string(numBlocks * numWarps) + ") {\n";

    // gen offset_src
    code += genOffset("offset_src", mappingSrc);
    code += genOffset("offset_dst", mappingDst);

    for (auto microOp : microOps) {
        code += microOp->generate();
    }
    code += "}\n}\n";
    return code;
}

std::string MetaGraph::genInvokeFunc() {
    std::vector<std::shared_ptr<MetaOp>> metaOps;
    for (auto &node : nodes) {
        metaOps.emplace_back(node.metaOps[0]);
    }
    std::string code = "";
    for (auto metaOp : metaOps) {
        code += metaOp->genInvokeFunc();
    }
    return code;
}

std::string MetaOp::genInvokeFunc() {
    std::string code = "";
    code += "void invoke_func_" + std::to_string(id) +
            "(float *src, float *dst) {\n";
    int numThreads = numWarps * 32;
    code += "dim3 gridDim(" + std::to_string(numBlocks) + ", 1);";
    code += "dim3 blockDim(" + std::to_string(numThreads) + ", 1);";
    code += "kernel_func<<<gridDim, blockDim>>>(src, dst);\n";
    code += "cudaCheckError();\n";
    code += "}\n";
    return code;
}

} // namespace memb