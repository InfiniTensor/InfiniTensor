#include "core/occamy_planner.h"
#include <algorithm>
#include <cassert>
#include <climits>

namespace infini {

size_t OccamyPlanner::alignUp(size_t size, size_t alignment) {
    return ((size + alignment - 1) / alignment) * alignment;
}

/*
 * Tensor Liveness Analyzer
 * 扫描 ops，为每个 isOthers() 的中间 tensor
 * 记录拓扑序下的firstUseStep/lastUseStep
 */
std::vector<TensorLiveness> OccamyPlanner::analyzeLiveness(const OpVec &ops) {
    std::unordered_map<TensorObj *, TensorLiveness> table;

    for (int step = 0; step < (int)ops.size(); ++step) {
        const auto &op = ops[step];

        for (const auto &t : op->getOutputs()) {
            if (!t || !t->isOthers())
                continue;
            // 第一次见到这个 tensor，创建 entry
            if (table.find(t.get()) == table.end()) {
                table[t.get()] = {
                    t.get(), alignUp(t->getBytes()),
                    step, // firstUseStep = 产生它的 op
                    step  // lastUseStep 先初始化为自己，后续更新
                };
            }
        }

        // 遍历 inputs：更新 lastUseStep
        for (const auto &t : op->getInputs()) {
            if (!t || !t->isOthers())
                continue;
            auto it = table.find(t.get());
            if (it != table.end()) {
                it->second.lastUseStep = step; // 每次作为 input 就更新 last use
            } else {
                // 如果 input tensor 没有 entry，说明它是用户显示创建的输入张量
                table[t.get()] = {t.get(), alignUp(t->getBytes()), step, step};
            }
        }
    }

    std::vector<TensorLiveness> result;
    result.reserve(table.size());
    for (auto &[ptr, entry] : table)
        result.push_back(entry);
    return result;
}

int OccamyPlanner::findFirstFit(const std::vector<OccamyFreeBlock> &freePool,
                                size_t size) {
    for (int i = 0; i < (int)freePool.size(); ++i) {
        if (freePool[i].end - freePool[i].start >= size)
            return i;
    }
    return -1;
}

int OccamyPlanner::findBestFit(const std::vector<OccamyFreeBlock> &freePool,
                               size_t size) {
    size_t bestSize = SIZE_MAX;
    int bestIdx = -1;
    for (int i = 0; i < (int)freePool.size(); ++i) {
        size_t freeSize = freePool[i].end - freePool[i].start;
        if (freeSize >= size && freeSize < bestSize) {
            bestSize = freeSize;
            bestIdx = i;
        }
    }
    return bestIdx;
}

// 在 freePool 中找到合适的位置，切割并插入 allocPool
size_t OccamyPlanner::allocSimul(TensorObj *tensor, size_t size,
                                 std::vector<OccamyFreeBlock> &freePool,
                                 std::vector<OccamyAllocBlock> &allocPool,
                                 Policy fitPolicy) {
    int fi = (fitPolicy == Policy::BEST_FIT) ? findBestFit(freePool, size)
                                             : findFirstFit(freePool, size);
    assert(fi >= 0 && "OccamyPlanner: pool exhausted during simulation");

    size_t startAddr = freePool[fi].start;

    // 插入 allocPool 按 start 地址排序
    OccamyAllocBlock blk = {tensor, startAddr, startAddr + size};
    auto insertPos = std::lower_bound(
        allocPool.begin(), allocPool.end(), blk,
        [](const OccamyAllocBlock &a, const OccamyAllocBlock &b) {
            return a.start < b.start;
        });
    allocPool.insert(insertPos, blk);

    // 从 freePool 切割（对应 Occamy freePool[fi].start_addr += mallocSize）
    freePool[fi].start += size;
    if (freePool[fi].start >= freePool[fi].end) {
        freePool.erase(freePool.begin() + fi);
    }

    return startAddr;
}

// 将 tensor 的空间归还 freePool，并合并相邻空闲块
bool OccamyPlanner::deallocSimul(TensorObj *tensor,
                                 std::vector<OccamyFreeBlock> &freePool,
                                 std::vector<OccamyAllocBlock> &allocPool) {
    // 找到 allocPool 中对应的块
    for (int ai = 0; ai < (int)allocPool.size(); ++ai) {
        if (allocPool[ai].tensorPtr != tensor)
            continue;

        OccamyAllocBlock freed = allocPool[ai];
        allocPool.erase(allocPool.begin() + ai);

        // 将 [freed.start, freed.end) 插回 freePool，并合并相邻块
        OccamyFreeBlock newFree = {freed.start, freed.end};

        bool merged = false;
        for (int fi = 0; fi < (int)freePool.size(); ++fi) {
            // 前向合并：newFree 紧接在 freePool[fi] 之后
            if (freePool[fi].end == newFree.start) {
                freePool[fi].end = newFree.end;
                // 后向合并：检查下一块是否紧接
                if (fi + 1 < (int)freePool.size() &&
                    freePool[fi].end == freePool[fi + 1].start) {
                    freePool[fi].end = freePool[fi + 1].end;
                    freePool.erase(freePool.begin() + fi + 1);
                }
                merged = true;
                break;
            }
            // 后向合并：newFree 紧接在 freePool[fi] 之前
            if (freePool[fi].start == newFree.end) {
                freePool[fi].start = newFree.start;
                if (fi > 0 && freePool[fi - 1].end == freePool[fi].start) {
                    freePool[fi - 1].end = freePool[fi].end;
                    freePool.erase(freePool.begin() + fi);
                }
                merged = true;
                break;
            }
        }
        // 如果没有合并，即是新的独立空闲块
        if (!merged) {
            auto pos = std::lower_bound(
                freePool.begin(), freePool.end(), newFree,
                [](const OccamyFreeBlock &a, const OccamyFreeBlock &b) {
                    return a.start < b.start;
                });
            freePool.insert(pos, newFree);
        }
        return true;
    }
    return false;
}

// 取当前 freePool 中最后一个有效块的 start 作为已使用峰值
size_t
OccamyPlanner::getCurrentPeak(const std::vector<OccamyFreeBlock> &freePool) {
    if (freePool.empty())
        return 0;
    return freePool.back().start;
}

size_t OccamyPlanner::runSchedule(
    const std::vector<TensorLiveness> &liveness, const OpVec &ops,
    Policy policy, std::unordered_map<TensorObj *, size_t> &tensorToOffset) {

    // 对 liveness 表排序（对应 LONGER_FIRST 和 BIGGER_FIRST 策略）
    std::vector<TensorLiveness> sortedLiveness = liveness;
    if (policy == Policy::LONGER_FIRST_FIT) {
        std::sort(sortedLiveness.begin(), sortedLiveness.end(),
                  [](const TensorLiveness &a, const TensorLiveness &b) {
                      int lenA = a.lastUseStep - a.firstUseStep;
                      int lenB = b.lastUseStep - b.firstUseStep;
                      return lenA > lenB;
                  });
    } else if (policy == Policy::BIGGER_FIRST_FIT) {
        std::sort(sortedLiveness.begin(), sortedLiveness.end(),
                  [](const TensorLiveness &a, const TensorLiveness &b) {
                      return a.sizeBytes > b.sizeBytes;
                  });
    }

    // 初始化 freePool
    std::vector<OccamyFreeBlock> freePool = {{0, SIZE_MAX / 2}};
    std::vector<OccamyAllocBlock> allocPool;
    tensorToOffset.clear();
    size_t peak = 0;

    if (policy == Policy::BIGGER_FIRST_FIT ||
        policy == Policy::LONGER_FIRST_FIT) {
        // 按排序顺序逐个处理张量
        // 每个张量分配时，构建一个"虚拟 freePool"：
        //   只有与当前张量生命周期重叠的已分配张量才占用空间
        //   不重叠的已分配张量的空间视为可用
        struct PlannedAlloc {
            TensorObj *tensor;
            size_t offset;
            size_t size;
            int firstUseStep;
            int lastUseStep;
        };
        std::vector<PlannedAlloc> planned;

        for (const auto &entry : sortedLiveness) {
            // 构建虚拟 freePool：只扣除与当前张量生命周期重叠的已分配张量
            std::vector<OccamyFreeBlock> virtualFreePool = {{0, SIZE_MAX / 2}};

            for (const auto &p : planned) {
                bool overlaps = !(p.lastUseStep < entry.firstUseStep ||
                                  p.firstUseStep > entry.lastUseStep);
                if (overlaps) {
                    // 从 virtualFreePool 中扣除 [p.offset, p.offset+p.size)
                    std::vector<OccamyFreeBlock> newFreePool;
                    for (const auto &fb : virtualFreePool) {
                        size_t occStart = p.offset;
                        size_t occEnd = p.offset + p.size;
                        if (fb.end <= occStart || fb.start >= occEnd) {
                            newFreePool.push_back(fb);
                        } else {
                            if (fb.start < occStart) {
                                newFreePool.push_back({fb.start, occStart});
                            }
                            if (fb.end > occEnd) {
                                newFreePool.push_back({occEnd, fb.end});
                            }
                        }
                    }
                    virtualFreePool = std::move(newFreePool);
                }
            }

            // 在虚拟 freePool 中 first-fit
            int fi = findFirstFit(virtualFreePool, entry.sizeBytes);
            assert(fi >= 0 && "OccamyPlanner: virtual pool exhausted");
            size_t offset = virtualFreePool[fi].start;

            tensorToOffset[entry.tensor] = offset;
            planned.push_back({entry.tensor, offset, entry.sizeBytes,
                               entry.firstUseStep, entry.lastUseStep});
        }

        // 峰值 = 所有张量中 (offset + size) 的最大值
        for (const auto &p : planned) {
            peak = std::max(peak, p.offset + p.size);
        }
    } else {
        // FIRST_FIT / BEST_FIT：严格按 topo 顺序 alloc/free
        std::unordered_map<int, std::vector<TensorLiveness *>> toAlloc, toFree;
        for (auto &entry : sortedLiveness) {
            toAlloc[entry.firstUseStep].push_back(&entry);
            toFree[entry.lastUseStep].push_back(&entry);
        }

        for (int step = 0; step < (int)ops.size(); ++step) {
            // 先 alloc 本 step 产生的 tensor（output）
            if (toAlloc.count(step)) {
                for (auto *entry : toAlloc[step]) {
                    size_t offset = allocSimul(entry->tensor, entry->sizeBytes,
                                               freePool, allocPool, policy);
                    tensorToOffset[entry->tensor] = offset;
                    peak = std::max(peak, getCurrentPeak(freePool));
                }
            }
            // 再 free 本 step 最后使用的 tensor（input 消耗完毕）
            if (toFree.count(step)) {
                for (auto *entry : toFree[step]) {
                    deallocSimul(entry->tensor, freePool, allocPool);
                }
            }
        }
    }

    return peak;
}

size_t
OccamyPlanner::plan(const OpVec &ops,
                    std::unordered_map<TensorObj *, size_t> &tensorToOffset) {
    /*
     * 四种显存规划策略：
     * 1. FIRST_FIT: 按拓扑顺序，第一个满足的空闲块
     * 2. BEST_FIT: 按拓扑顺序，选择最小的满足块
     * 3. LONGER_FIRST_FIT: 按生命周期降序，使用 first-fit
     * 4. BIGGER_FIRST_FIT: 按大小降序，使用 first-fit
     */
    const std::vector<Policy> policies = {
        Policy::FIRST_FIT,
        Policy::BEST_FIT,
        Policy::LONGER_FIRST_FIT,
        Policy::BIGGER_FIRST_FIT,
    };

    size_t bestPeak = SIZE_MAX;
    std::unordered_map<TensorObj *, size_t> bestOffsets;

    // 张量的 liveness 分析
    auto liveness = analyzeLiveness(ops);

    for (auto policy : policies) {
        std::unordered_map<TensorObj *, size_t> offsets;
        size_t peak = runSchedule(liveness, ops, policy, offsets);
        if (peak < bestPeak) {
            bestPeak = peak;
            bestOffsets = std::move(offsets);
        }
    }

    tensorToOffset = std::move(bestOffsets);
    return bestPeak; // 返回最小峰值内存大小（字节）
}

} // namespace infini
