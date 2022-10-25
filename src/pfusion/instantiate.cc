#include "pfusion/instantiate.h"
#include "pfusion/micro_kernel/binary.h"
#include "pfusion/micro_kernel/memory.h"
#include "pfusion/micro_kernel/unary.h"
#include "pfusion/micro_op.h"

namespace memb {

size_t getSize(const std::vector<int> &shape) {
    size_t size = 1;
    for (auto x : shape) {
        size *= x;
    }
    return size;
}

std::vector<std::shared_ptr<MetaOp>>
instantiateUnary(const OpType opType,
                 std::vector<std::shared_ptr<Pointer>> ptrs,
                 const std::vector<int> &shape) {
    std::vector<std::shared_ptr<MetaOp>> metaOps;
    size_t size = getSize(shape);

    auto metaOp = std::make_shared<MetaOp>();
    metaOp->main_loop_st = 0;
    metaOp->main_loop_ed = size / 32 / 8;
    metaOp->numBlocks = 108;
    metaOp->numWarps = 8;
    metaOp->numReg = 8;
    metaOp->numSmem = 0;

    metaOp->mappingSrc = std::make_shared<TensorMapping>();
    metaOp->mappingSrc->shape = {32 * 8, int(size / 32 / 8)};
    metaOp->mappingSrc->map = {1};

    metaOp->mappingDst = std::make_shared<TensorMapping>();
    metaOp->mappingDst->shape = {32 * 8, int(size / 32 / 8)};
    metaOp->mappingDst->map = {1};

    metaOp->ptrs = ptrs;
    auto buf = Pointer::buildPtr(REG, "buf", "inst_idx");

    metaOp->microOps.emplace_back(std::make_shared<MemoryOp>(
        READ, Pointer::buildPtr(ptrs[0], "offset + inst_idx * 32 + lane_id"),
        buf, 8, 32));
    metaOp->microOps.emplace_back(
        std::make_shared<UnaryOp>(opType, buf, buf, 8, 32));

    metaOp->microOps.emplace_back(std::make_shared<MemoryOp>(
        WRITE, buf,
        Pointer::buildPtr(ptrs[1], "offset + inst_idx * 32 + lane_id"), 8, 32));

    metaOps.emplace_back(metaOp);
    return metaOps;
}

std::vector<std::shared_ptr<MetaOp>>
instantiateBinary(const OpType opType,
                  std::vector<std::shared_ptr<Pointer>> ptrs,
                  const std::vector<int> &shape) {
    std::vector<std::shared_ptr<MetaOp>> metaOps;
    size_t size = getSize(shape);

    auto metaOp = std::make_shared<MetaOp>();
    metaOp->main_loop_st = 0;
    metaOp->main_loop_ed = size / 32 / 8;
    metaOp->numBlocks = 108;
    metaOp->numWarps = 8;
    metaOp->numReg = 24;
    metaOp->numSmem = 0;

    metaOp->mappingSrc = std::make_shared<TensorMapping>();
    metaOp->mappingSrc->shape = {32 * 8, int(size / 32 / 8)};
    metaOp->mappingSrc->map = {1};

    metaOp->mappingDst = std::make_shared<TensorMapping>();
    metaOp->mappingDst->shape = {32 * 8, int(size / 32 / 8)};
    metaOp->mappingDst->map = {1};

    metaOp->ptrs = ptrs;
    auto buf0 = Pointer::buildPtr(REG, "buf", "inst_idx");
    auto buf1 = Pointer::buildPtr(REG, "buf", "inst_idx + 8");
    auto buf2 = Pointer::buildPtr(REG, "buf", "inst_idx + 16");

    metaOp->microOps.emplace_back(std::make_shared<MemoryOp>(
        READ, Pointer::buildPtr(ptrs[0], "offset + inst_idx * 32 + lane_id"),
        buf0, 8, 32));
    metaOp->microOps.emplace_back(std::make_shared<MemoryOp>(
        READ, Pointer::buildPtr(ptrs[1], "offset + inst_idx * 32 + lane_id"),
        buf1, 8, 32));
    metaOp->microOps.emplace_back(
        std::make_shared<BinaryOp>(opType, buf0, buf1, buf2, 8, 32));
    metaOp->microOps.emplace_back(std::make_shared<MemoryOp>(
        WRITE, buf2,
        Pointer::buildPtr(ptrs[2], "offset + inst_idx * 32 + lane_id"), 8, 32));

    metaOps.emplace_back(metaOp);
    return metaOps;
}

std::vector<std::shared_ptr<MetaOp>> instantiateTranspose(
    const OpType opType, std::vector<std::shared_ptr<Pointer>> ptrs,
    const std::vector<int> &shape, const std::vector<int> &perm) {
    std::vector<std::shared_ptr<MetaOp>> metaOps;

    size_t size = 1;
    for (auto x : shape) {
        size *= x;
    }

    auto metaOp = std::make_shared<MetaOp>();
    metaOp->mappingSrc = std::make_shared<TensorMapping>();
    metaOp->mappingSrc->shape.resize(shape.size());
    metaOp->mappingSrc->map.clear();
    size_t parallelSize = 1;
    for (size_t i = 0; i < shape.size(); i++) {
        metaOp->mappingSrc->shape[perm[i]] = shape[i];
        if (i != 0 && perm[i] != 0) {
            metaOp->mappingSrc->map.emplace_back(perm[i]);
            parallelSize *= shape[i];
        }
    }
    metaOp->mappingDst = std::make_shared<TensorMapping>();
    metaOp->mappingDst->shape.clear();
    metaOp->mappingDst->map.clear();
    for (size_t i = 0; i < shape.size(); i++) {
        metaOp->mappingDst->shape.emplace_back(shape[i]);
        if (i != 0 && perm[i] != 0) {
            metaOp->mappingDst->map.emplace_back(i);
        }
    }

    metaOp->main_loop_st = 0;
    metaOp->main_loop_ed = parallelSize;
    metaOp->numBlocks = 108;
    metaOp->numWarps = 8;

    int numTileA = (shape[perm[0]] - 1) / 32 + 1;
    int numTileB = (shape[0] - 1) / 32 + 1;
    // std::cout << numTileA << " " << numTileB << std::endl;

    metaOp->numReg = 32;
    metaOp->numSmem = 32 * 33;

    size_t stride_src = 1, stride_dst = 1;
    for (size_t i = 0; i < shape.size(); i++) {
        if (perm[i] < perm[0]) {
            stride_src *= shape[i];
        }
    }
    for (size_t i = 0; perm[i] != 0; i++) {
        stride_dst *= shape[i];
    }

    // TODO: tiling is a metaOp or microOps?

    metaOp->ptrs = ptrs;
    auto smem = Pointer::buildPtr(SRAM, "smem", "warp_id * 32 * 33");
    auto buf = Pointer::buildPtr(REG, "buf", "inst_idx");

    for (int i = 0; i < numTileA; i++) {
        for (int j = 0; j < numTileB; j++) {
            auto src_ptr = Pointer::buildPtr(
                ptrs[0], "offset_src + " +
                             std::to_string(j * 32 * stride_src + i * 32) +
                             "+" + "inst_idx * " + std::to_string(stride_src) +
                             " + lane_id");
            metaOp->microOps.emplace_back(std::make_shared<MemoryOp>(
                READ, src_ptr, buf, std::min(32, shape[0]),
                std::min(32, shape[perm[0]])));
            metaOp->microOps.emplace_back(std::make_shared<MemoryOp>(
                WRITE, buf, Pointer::buildPtr(smem, "inst_idx * 33 + lane_id"),
                std::min(32, shape[0]), std::min(32, shape[perm[0]])));
            metaOp->microOps.emplace_back(std::make_shared<MemoryOp>(
                READ, Pointer::buildPtr(smem, "lane_id * 33 + inst_idx"), buf,
                std::min(32, shape[perm[0]]), std::min(32, shape[0])));
            auto dst_ptr = Pointer::buildPtr(
                ptrs[1], "offset_dst + " +
                             std::to_string(i * 32 * stride_dst + j * 32) +
                             "+" + "inst_idx * " + std::to_string(stride_dst) +
                             " + lane_id");
            metaOp->microOps.emplace_back(std::make_shared<MemoryOp>(
                WRITE, buf, dst_ptr, std::min(32, shape[perm[0]]),
                std::min(32, shape[0])));
        }
    }
    metaOps.emplace_back(metaOp);

    return metaOps;
}

} // namespace memb
