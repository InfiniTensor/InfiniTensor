#include "pfusion/meta_op.h"
#include "pfusion/micro_kernel/binary.h"
#include "pfusion/micro_kernel/memory.h"

namespace memb {

std::string TensorMapping::genOffset() {
    std::string code = "int " + offset() + " = 0;\n";

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

void MetaOp::optimize() {
    if (microOps.size() == 0)
        return;
    std::vector<std::shared_ptr<MicroOp>> ops;
    int numOp = microOps.size();
    int cur = 0;
    for (int i = 1; i < numOp; i++) {
        auto next = MicroOp::merge(microOps[cur], microOps[i]);
        if (next == nullptr) {
            ops.emplace_back(microOps[cur]);
            cur = i;
        } else {
            cur = microOps.size();
            microOps.emplace_back(next);
        }
    }
    ops.emplace_back(microOps[cur]);
    microOps.clear();
    std::unordered_set<std::string> ptrSet;
    for (auto op : ops) {
        for (auto ptr : op->getPtrs()) {
            ptrSet.emplace(ptr->getName());
        }
        if (op->getType() != EMPTY) {
            microOps.emplace_back(op);
        }
    }
    std::vector<std::shared_ptr<Pointer>> newPtrs;
    for (auto ptr : ptrs) {
        if (ptrSet.find(ptr->getName()) != ptrSet.end()) {
            newPtrs.emplace_back(ptr);
        }
    }
    ptrs.clear();
    for (auto ptr : newPtrs) {
        ptrs.emplace_back(ptr);
    }
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
    code += "int lane_id = threadIdx.x % " + std::to_string(numLanes) + ";\n";
    code += "int warp_id = threadIdx.x / " + std::to_string(numLanes) + ";\n";
    code += "int parallel_idx = blockIdx.x * " + std::to_string(numGroups) +
            " + warp_id;\n";
    if (numReg != 0) {
        code += "float buf[" + std::to_string(numReg) + "];\n";
    }
    if (numSmem != 0) {
        code += "__shared__ float smem[" + std::to_string(numSmem * numGroups) +
                "];\n";
    }

    code += "for (int loop_idx = parallel_idx; loop_idx < " +
            std::to_string(main_loop_ed) +
            "; loop_idx += " + std::to_string(numBlocks * numGroups) + ") {\n";

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
    int numThreads = numGroups * numLanes;
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

std::shared_ptr<MetaOp> MetaOp::merge(std::shared_ptr<MetaOp> metaOp0,
                                      std::shared_ptr<MetaOp> metaOp1) {
    IT_ASSERT(metaOp0->checkValid());
    IT_ASSERT(metaOp1->checkValid());
    // Check unmergeable
    if (metaOp0->main_loop_st != metaOp1->main_loop_st ||
        metaOp0->main_loop_ed != metaOp1->main_loop_ed ||
        metaOp0->numBlocks != metaOp1->numBlocks ||
        metaOp0->numGroups != metaOp1->numGroups ||
        metaOp0->numLanes != metaOp1->numLanes) {
        return nullptr;
    }
    auto metaOp = std::make_shared<MetaOp>();
    metaOp->main_loop_st = metaOp0->main_loop_st;
    metaOp->main_loop_ed = metaOp0->main_loop_ed;
    metaOp->numBlocks = metaOp0->numBlocks;
    metaOp->numGroups = metaOp0->numGroups;
    metaOp->numLanes = metaOp0->numLanes;
    metaOp->numReg = metaOp0->numReg + metaOp1->numReg;
    metaOp->numSmem = metaOp0->numSmem + metaOp1->numSmem;

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

std::shared_ptr<MetaOp> MetaOp::buildBiasOp(const std::vector<size_t> &shape) {
    IT_ASSERT(shape.size() == 2);
    auto metaOp = std::make_shared<MetaOp>();
    metaOp->main_loop_st = 0;
    metaOp->main_loop_ed = shape[1] * (shape[0] / 32 / 4);
    metaOp->numBlocks = 80;
    metaOp->numGroups = 4;
    metaOp->numLanes = 32;
    metaOp->numReg = 4;
    metaOp->numSmem = 0;

    metaOp->mappings.emplace_back(TensorMapping::build(
        std::string("input"),
        std::vector<size_t>({32 * 4, (shape[0] - 1) / (32 * 4) + 1, shape[1]}),
        std::vector<size_t>({1, 32 * 4, shape[0]}),
        std::vector<size_t>({1, 2})));
    metaOp->mappings.emplace_back(TensorMapping::buildWithMap(
        std::string("bias"), std::vector<size_t>({shape[0], shape[1]}),
        std::vector<size_t>({1})));

    metaOp->ptrs = std::vector<std::shared_ptr<Pointer>>();
    auto &ptrs = metaOp->ptrs;
    ptrs.emplace_back(Pointer::buildPtr(DRAM, "input"));
    ptrs.emplace_back(Pointer::buildPtr(DRAM, "bias"));
    ptrs.emplace_back(Pointer::buildPtr(DRAM, "output"));

    auto buf_input = Pointer::buildPtr(REG, "buf", "inst_idx");
    auto buf_bias = Pointer::buildPtr(REG, "buf", "4");
    auto buf_output = Pointer::buildPtr(REG, "buf", "inst_idx");

    // @cond group_id * 4 * 32 + inst_idx * 32 + lane_id < shape[0]
    metaOp->microOps.emplace_back(MemoryOp::build(
        READ,
        Pointer::buildPtr(ptrs[0], "offset_input + inst_idx * 32 + lane_id"),
        buf_input, 4, 32));
    metaOp->microOps.emplace_back(MemoryOp::build(
        READ, Pointer::buildPtr(ptrs[1], "offset_bias"), buf_bias, 1, 32));
    metaOp->microOps.emplace_back(std::make_shared<BinaryOp>(
        ADD, buf_input, buf_bias, buf_output, 4, 32));
    // @cond group_id * 4 * 32 + inst_idx * 32 + lane_id < shape[0]
    metaOp->microOps.emplace_back(MemoryOp::build(
        WRITE, buf_output,
        Pointer::buildPtr(ptrs[1], "offset_input + inst_idx * 32 + lane_id"), 8,
        32));
    return metaOp;
}

std::shared_ptr<MetaOp>
MetaOp::buildTransposeOp(const std::vector<size_t> &shape,
                         const std::vector<size_t> &perm) {
    IT_ASSERT(perm[0] == 0 && shape[0] >= 32);
    IT_ASSERT(shape.size() == 3);
    auto metaOp = std::make_shared<MetaOp>();
    size_t numInst, extraDim;
    std::vector<size_t> map_shape, map_stride;
    if (shape[0] <= 4 * 32) {
        numInst = (shape[0] - 1) / 32 + 1;
        extraDim = 1;
        metaOp->mappings.emplace_back(TensorMapping::build(
            std::string("input"),
            std::vector<size_t>({shape[0], shape[perm[1]], shape[perm[2]]}),
            std::vector<size_t>({1, shape[0], shape[0] * shape[perm[1]]}),
            std::vector<size_t>({perm[1], perm[2]})));
        metaOp->mappings.emplace_back(TensorMapping::build(
            std::string("output"),
            std::vector<size_t>({shape[0], shape[1], shape[2]}),
            std::vector<size_t>({1, shape[0], shape[0] * shape[1]}),
            std::vector<size_t>({1, 2})));
        // cond: local_id < shape[0];
    } else {
        numInst = 4;
        extraDim = (shape[0] - 1) / 128 + 1;
        metaOp->mappings.emplace_back(TensorMapping::build(
            std::string("input"),
            std::vector<size_t>(
                {128, extraDim, shape[perm[1]], shape[perm[2]]}),
            std::vector<size_t>({1, 128, shape[0], shape[0] * shape[perm[1]]}),
            std::vector<size_t>({1, perm[1] + 1, perm[2] + 1})));
        metaOp->mappings.emplace_back(TensorMapping::build(
            std::string("output"),
            std::vector<size_t>({128, extraDim, shape[1], shape[2]}),
            std::vector<size_t>({1, 128, shape[0], shape[0] * shape[1]}),
            std::vector<size_t>({1, 2, 3})));
        // cond loop_idx % extraDim * 128 + local_id < shape[0];
    }
    metaOp->main_loop_st = 0;
    metaOp->main_loop_ed = shape[1] * shape[2] * extraDim;
    metaOp->numBlocks = 80;
    metaOp->numGroups = 4;
    metaOp->numLanes = 32;
    metaOp->numReg = 4;
    metaOp->numSmem = 0;

    metaOp->ptrs = std::vector<std::shared_ptr<Pointer>>();
    auto &ptrs = metaOp->ptrs;
    ptrs.emplace_back(Pointer::buildPtr(DRAM, "input"));
    ptrs.emplace_back(Pointer::buildPtr(DRAM, "output"));

    auto buf_input = Pointer::buildPtr(REG, "buf", "inst_idx");
    auto buf_output = Pointer::buildPtr(REG, "buf", "inst_idx");

    // @cond group_id * 4 * 32 + inst_idx * 32 + lane_id < shape[0]
    std::vector<size_t> cond = {shape[0], extraDim, 128};
    auto inPtr =
        Pointer::buildPtr(ptrs[0], "offset_input + inst_idx * 32 + lane_id");
    auto opRead = MemoryOp::build(READ, inPtr, buf_input, numInst, 32, cond);
    auto outPtr =
        Pointer::buildPtr(ptrs[1], "offset_output + inst_idx * 32 + lane_id");
    auto opWrite =
        MemoryOp::build(WRITE, buf_output, outPtr, numInst, 32, cond);
    metaOp->microOps = std::vector<std::shared_ptr<MicroOp>>({opRead, opWrite});

    return metaOp;
}

} // namespace memb