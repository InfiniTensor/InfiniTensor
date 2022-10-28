#include "pfusion/instantiate.h"
#include "pfusion/meta_op.h"
#include "pfusion/micro_kernel/binary.h"
#include "pfusion/micro_kernel/memory.h"
#include "pfusion/micro_kernel/unary.h"
#include "pfusion/micro_op.h"

namespace memb {

size_t getSize(const std::vector<size_t> &shape) {
    size_t size = 1;
    for (auto x : shape) {
        size *= x;
    }
    return size;
}

size_t min(size_t a, size_t b) { return (a < b) ? a : b; }

std::vector<std::shared_ptr<MetaOp>>
instantiateUnary(const OpType opType,
                 std::vector<std::shared_ptr<Pointer>> ptrs,
                 const std::vector<size_t> &shape) {
    std::vector<std::shared_ptr<MetaOp>> metaOps;
    size_t size = getSize(shape);

    auto metaOp = std::make_shared<MetaOp>();
    metaOp->main_loop_st = 0;
    metaOp->main_loop_ed = size / 32 / 8;
    metaOp->numBlocks = 108;
    metaOp->numWarps = 8;
    metaOp->numReg = 8;
    metaOp->numSmem = 0;

    metaOp->mappings.emplace_back(std::make_shared<TensorMapping>(
        std::string("src"), std::vector<size_t>({32 * 8, size / 32 / 8}),
        std::vector<size_t>({1})));

    metaOp->ptrs = ptrs;
    auto buf = Pointer::buildPtr(REG, "buf", "inst_idx");

    metaOp->microOps.emplace_back(std::make_shared<MemoryOp>(
        READ,
        Pointer::buildPtr(ptrs[0], "offset_src + inst_idx * 32 + lane_id"), buf,
        8, 32));
    metaOp->microOps.emplace_back(
        std::make_shared<UnaryOp>(opType, buf, buf, 8, 32));

    metaOp->microOps.emplace_back(std::make_shared<MemoryOp>(
        WRITE, buf,
        Pointer::buildPtr(ptrs[1], "offset_src + inst_idx * 32 + lane_id"), 8,
        32));

    metaOps.emplace_back(metaOp);
    return metaOps;
}

std::vector<std::shared_ptr<MetaOp>>
instantiateBinary(const OpType opType,
                  std::vector<std::shared_ptr<Pointer>> ptrs,
                  const std::vector<size_t> &shape) {
    std::vector<std::shared_ptr<MetaOp>> metaOps;
    size_t size = getSize(shape);

    auto metaOp = std::make_shared<MetaOp>();
    metaOp->main_loop_st = 0;
    metaOp->main_loop_ed = size / 32 / 8;
    metaOp->numBlocks = 108;
    metaOp->numWarps = 8;
    metaOp->numReg = 24;
    metaOp->numSmem = 0;

    metaOp->mappings.emplace_back(std::make_shared<TensorMapping>(
        std::string("src"), std::vector<size_t>({32 * 8, size / 32 / 8}),
        std::vector<size_t>({1})));

    metaOp->ptrs = ptrs;
    auto buf0 = Pointer::buildPtr(REG, "buf", "inst_idx");
    auto buf1 = Pointer::buildPtr(REG, "buf", "inst_idx + 8");
    auto buf2 = Pointer::buildPtr(REG, "buf", "inst_idx + 16");

    metaOp->microOps.emplace_back(std::make_shared<MemoryOp>(
        READ,
        Pointer::buildPtr(ptrs[0], "offset_src + inst_idx * 32 + lane_id"),
        buf0, 8, 32));
    metaOp->microOps.emplace_back(std::make_shared<MemoryOp>(
        READ,
        Pointer::buildPtr(ptrs[1], "offset_src + inst_idx * 32 + lane_id"),
        buf1, 8, 32));
    metaOp->microOps.emplace_back(
        std::make_shared<BinaryOp>(opType, buf0, buf1, buf2, 8, 32));
    metaOp->microOps.emplace_back(std::make_shared<MemoryOp>(
        WRITE, buf2,
        Pointer::buildPtr(ptrs[2], "offset_src + inst_idx * 32 + lane_id"), 8,
        32));

    metaOps.emplace_back(metaOp);
    return metaOps;
}

std::vector<std::shared_ptr<MetaOp>> instantiateTranspose(
    const OpType opType, std::vector<std::shared_ptr<Pointer>> ptrs,
    const std::vector<size_t> &shape, const std::vector<size_t> &perm) {
    std::vector<std::shared_ptr<MetaOp>> metaOps;

    size_t size = 1;
    for (auto x : shape) {
        size *= x;
    }

    auto metaOp = std::make_shared<MetaOp>();

    std::vector<size_t> srcShape(shape.size());
    std::vector<size_t> srcMap;
    size_t parallelSize = 1;
    for (size_t i = 0; i < shape.size(); i++) {
        srcShape[perm[i]] = shape[i];
        if (i != 0 && perm[i] != 0) {
            srcMap.emplace_back(perm[i]);
            parallelSize *= shape[i];
        }
    }
    metaOp->mappings.emplace_back(
        std::make_shared<TensorMapping>("src", srcShape, srcMap));

    std::vector<size_t> dstMap;
    for (size_t i = 0; i < shape.size(); i++) {
        if (i != 0 && perm[i] != 0) {
            dstMap.emplace_back(i);
        }
    }
    metaOp->mappings.emplace_back(
        std::make_shared<TensorMapping>("dst", shape, dstMap));

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
                READ, src_ptr, buf, min(32u, shape[0]),
                min(32, shape[perm[0]])));
            metaOp->microOps.emplace_back(std::make_shared<MemoryOp>(
                WRITE, buf, Pointer::buildPtr(smem, "inst_idx * 33 + lane_id"),
                min(32, shape[0]), min(32, shape[perm[0]])));
            metaOp->microOps.emplace_back(std::make_shared<MemoryOp>(
                READ, Pointer::buildPtr(smem, "lane_id * 33 + inst_idx"), buf,
                min(32, shape[perm[0]]), min(32, shape[0])));
            auto dst_ptr = Pointer::buildPtr(
                ptrs[1], "offset_dst + " +
                             std::to_string(i * 32 * stride_dst + j * 32) +
                             "+" + "inst_idx * " + std::to_string(stride_dst) +
                             " + lane_id");
            metaOp->microOps.emplace_back(std::make_shared<MemoryOp>(
                WRITE, buf, dst_ptr, min(32, shape[perm[0]]),
                min(32, shape[0])));
        }
    }
    metaOps.emplace_back(metaOp);

    return metaOps;
}

std::vector<std::shared_ptr<MetaOp>>
instantiateGather(const OpType opType,
                  const std::vector<std::shared_ptr<Pointer>> &ptrs,
                  const std::vector<size_t> &inputShape,
                  const std::vector<size_t> &indexShape,
                  const std::vector<size_t> &outputShape, const size_t axis) {
    std::vector<std::shared_ptr<MetaOp>> metaOps;
    size_t par_size = 1;
    for (size_t i = 0; i < outputShape.size() - 1; i++) {
        par_size *= inputShape[i];
    }
    size_t seq_size = inputShape[outputShape.size() - 1];

    auto metaOp = std::make_shared<MetaOp>();
    metaOp->main_loop_st = 0;
    metaOp->main_loop_ed = par_size;
    metaOp->numBlocks = 108;
    metaOp->numWarps = 2;
    metaOp->numReg = 24;
    metaOp->numSmem = 0;

    metaOp->mappings.emplace_back(std::make_shared<TensorMapping>(
        std::string("src"), std::vector<size_t>({seq_size, par_size}),
        std::vector<size_t>({1})));

    metaOp->ptrs = ptrs;
    auto buf0 = Pointer::buildPtr(REG, "buf", "inst_idx");
    auto buf1 = Pointer::buildPtr(REG, "buf", "inst_idx + 8");
    auto buf2 = Pointer::buildPtr(REG, "buf", "inst_idx + 16");

    metaOps.emplace_back(metaOp);
    return metaOps;
}

std::vector<std::shared_ptr<MetaOp>>
instantiateReduce(const OpType opType,
                  const std::vector<std::shared_ptr<Pointer>> &ptrs,
                  const std::vector<size_t> &inputShape, const size_t axis) {
    std::vector<std::shared_ptr<MetaOp>> metaOps;
    size_t par_size = 1;
    for (size_t i = 0; i < inputShape.size(); i++) {
        if (i != axis) {
            par_size *= inputShape[i];
        }
    }
    size_t seq_size = inputShape[axis];

    auto metaOp = std::make_shared<MetaOp>();
    metaOp->main_loop_st = 0;
    metaOp->main_loop_ed = par_size;
    metaOp->numBlocks = 108;
    metaOp->numWarps = 2;
    metaOp->numReg = 24;
    metaOp->numSmem = 0;

    metaOp->mappings.emplace_back(std::make_shared<TensorMapping>(
        std::string("src"), std::vector<size_t>({seq_size, par_size}),
        std::vector<size_t>({1})));

    metaOp->ptrs = ptrs;
    auto buf0 = Pointer::buildPtr(REG, "buf", "inst_idx");
    auto buf1 = Pointer::buildPtr(REG, "buf", "inst_idx + 8");
    auto buf2 = Pointer::buildPtr(REG, "buf", "inst_idx + 16");

    metaOps.emplace_back(metaOp);
    return metaOps;
}

std::vector<std::shared_ptr<MetaOp>>
instantiateBroadcast(const OpType opType,
                     const std::vector<std::shared_ptr<Pointer>> &ptrs,
                     const std::vector<size_t> &inputShape, const size_t axis,
                     const size_t num) {
    std::vector<std::shared_ptr<MetaOp>> metaOps;
    size_t par_size = getSize(inputShape);
    size_t seq_size = num;

    auto metaOp = std::make_shared<MetaOp>();
    metaOp->main_loop_st = 0;
    metaOp->main_loop_ed = par_size;
    metaOp->numBlocks = 108;
    metaOp->numWarps = 2;
    metaOp->numReg = 24;
    metaOp->numSmem = 0;

    metaOp->mappings.emplace_back(std::make_shared<TensorMapping>(
        std::string("src"), std::vector<size_t>({seq_size, par_size}),
        std::vector<size_t>({1})));

    metaOp->ptrs = ptrs;
    auto buf0 = Pointer::buildPtr(REG, "buf", "inst_idx");
    auto buf1 = Pointer::buildPtr(REG, "buf", "inst_idx + 8");
    auto buf2 = Pointer::buildPtr(REG, "buf", "inst_idx + 16");

    metaOps.emplace_back(metaOp);
    return metaOps;
}

} // namespace memb
