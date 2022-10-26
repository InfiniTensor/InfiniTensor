#include "pfusion/meta_op.h"

namespace memb {

std::string TensorMapping::genOffset() {
    std::string code = "int " + offset() + " = 0;\n";
    std::vector<int> stride;
    stride.emplace_back(1);
    for (size_t i = 1; i < shape.size(); i++) {
        stride.emplace_back(stride[i - 1] * shape[i - 1]);
    }

    std::string bufName = name + "_buf";
    code += "int " + bufName + " = loop_idx;\n";
    for (size_t i = 0; i < map.size(); i++) {
        code += name + " += " + bufName + " % " +
                std::to_string(shape[map[i]]) + " * " +
                std::to_string(stride[map[i]]) + ";\n";
        code += bufName + " /= " + std::to_string(shape[map[i]]) + ";\n";
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
    for (auto mapping : mappings) {
        code += mapping->genOffset();
    }

    for (auto microOp : microOps) {
        code += microOp->generate();
    }
    code += "}\n}\n";
    return code;
}

std::string MetaOp::genInvokeFunc() {
    std::string code = "";
    code += "void invoke_func_" + std::to_string(id) + "(";
    IT_ASSERT(ptrs.size() > 0);
    code += "float *" + ptrs[0]->getName();
    for (size_t i = 1; i < ptrs.size(); i++) {
        code += ", float *" + ptrs[i]->getName();
    }
    code += ") {\n";
    int numThreads = numWarps * 32;
    code += "dim3 gridDim(" + std::to_string(numBlocks) + ", 1);";
    code += "dim3 blockDim(" + std::to_string(numThreads) + ", 1);";
    code += "kernel_func_" + std::to_string(id) + "<<<gridDim, blockDim>>>(";
    IT_ASSERT(ptrs.size() > 0);
    code += ptrs[0]->getName();
    for (size_t i = 1; i < ptrs.size(); i++) {
        code += ", " + ptrs[i]->getName();
    }
    code += ");\n";
    code += "cudaCheckError();\n";
    code += "}\n";
    return code;
}

std::shared_ptr<MetaOp> MetaOp::buildByMerge(std::shared_ptr<MetaOp> metaOp0,
                                             std::shared_ptr<MetaOp> metaOp1) {
    IT_ASSERT(metaOp0->checkValid());
    IT_ASSERT(metaOp1->checkValid());
    // Check unmergeable
    if (metaOp0->main_loop_st != metaOp1->main_loop_st ||
        metaOp0->main_loop_ed != metaOp1->main_loop_ed ||
        metaOp0->numBlocks != metaOp1->numBlocks ||
        metaOp0->numReg != metaOp1->numReg ||
        metaOp0->numSmem != metaOp1->numSmem) {
        return nullptr;
    }
    auto metaOp = std::make_shared<MetaOp>();
    metaOp->main_loop_st = metaOp0->main_loop_st;
    metaOp->main_loop_ed = metaOp0->main_loop_ed;
    metaOp->numBlocks = metaOp0->numBlocks;
    metaOp->numReg = metaOp0->numReg;
    metaOp->numSmem = metaOp0->numSmem;

    // Merge ptr
    std::unordered_set<size_t> ptrSet;
    for (auto ptr : metaOp0->ptrs) {
        IT_ASSERT(ptrSet.find(ptr->getHash()) == ptrSet.end());
        metaOp->ptrs.emplace_back(ptr);
        ptrSet.emplace(ptr->getHash());
    }
    for (auto ptr : metaOp1->ptrs) {
        if (ptrSet.find(ptr->getHash()) == ptrSet.end()) {
            metaOp->ptrs.emplace_back(ptr);
            ptrSet.emplace(ptr->getHash());
        }
    }

    // Merge mapping
    std::unordered_set<size_t> mappingSet;
    for (auto mapping : metaOp0->mappings) {
        IT_ASSERT(mappingSet.find(mapping->getHash()) == mappingSet.end());
        metaOp->mappings.emplace_back(mapping);
        mappingSet.emplace(mapping->getHash());
    }
    for (auto mapping : metaOp1->mappings) {
        if (mappingSet.find(mapping->getHash()) == mappingSet.end()) {
            metaOp->mappings.emplace_back(mapping);
            mappingSet.emplace(mapping->getHash());
        }
    }

    // Merge microOps.
    // TODO: make it a graph.
    for (auto microOp : metaOp0->microOps) {
        metaOp->microOps.emplace_back(microOp);
    }
    for (auto microOp : metaOp1->microOps) {
        metaOp->microOps.emplace_back(microOp);
    }
    for (auto microOp : metaOp->microOps) {
        microOp->print();
    }
    // TODO: elimiate microOps.

    return metaOp;
}

} // namespace memb