#include "pfusion/instantiate.h"
#include "pfusion/common.h"
#include "pfusion/micro_kernel/memory.h"

namespace memb {
std::vector<std::shared_ptr<MetaOp>> instantiateAbs(std::vector<int> shape) {
    std::vector<std::shared_ptr<MetaOp>> metaOps;
    auto metaOp = std::make_shared<MetaOp>();
    size_t sz = 1;
    for (auto d : shape) {
        sz *= d;
    }
    metaOp->main_loop_st = 0;
    metaOp->main_loop_ed = sz / 1024;
    metaOp->parallelism = 1024;
    metaOp->main_loop_step = 32;
    auto sram_write = std::make_shared<MemoryOp>();
    sram_write->memoryType = MemoryOp::MemoryType::SRAM;
    sram_write->opType = MemoryOp::OpType::WRITE;
    sram_write->ptr = buildPtr("smem", "warp_id * 32 * 32 * 2");
    sram_write->num = "test";
    sram_write->offset = "inst_idx * 32 + lane_id";
    sram_write->reg = "inst_idx";
    metaOp->microOps.emplace_back(sram_write);
    metaOps.emplace_back(metaOp);
    return metaOps;
}

std::vector<std::shared_ptr<MetaOp>> instantiateRelu(std::vector<int> shape) {
    std::vector<std::shared_ptr<MetaOp>> metaOps;
    auto metaOp = std::make_shared<MetaOp>();
    size_t sz = 1;
    for (auto d : shape) {
        sz *= d;
    }
    metaOp->main_loop_st = 0;
    metaOp->main_loop_ed = sz / 1024;
    metaOp->parallelism = 1024;
    metaOp->main_loop_step = 32;
    auto sram_write = std::make_shared<MemoryOp>();
    sram_write->memoryType = MemoryOp::MemoryType::SRAM;
    sram_write->opType = MemoryOp::OpType::WRITE;
    sram_write->ptr = buildPtr("smem", "warp_id * 32 * 32 * 2");
    sram_write->num = "test";
    sram_write->offset = "inst_idx * 32 + lane_id";
    sram_write->reg = "inst_idx";
    metaOp->microOps.emplace_back(sram_write);
    metaOps.emplace_back(metaOp);
    return metaOps;
}

std::vector<std::shared_ptr<MetaOp>>
instantiateTranspose(std::vector<int> shape, std::vector<int> perm) {
    return {};
}

} // namespace memb
