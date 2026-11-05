#pragma once
#include "core/runtime.h"
#include <cstddef>
#include <memory>

namespace infini {
class ConvObj;
class CudaRuntimeObj;

bool convNhwcEnabled();
bool convNhwcRegionEnabled();
bool convNhwcEligible(const Ref<ConvObj> &op);
// Metadata-only discovery follows data edges, independent of execution order.
std::vector<OpVec> findConvNhwcRegions(const OpVec &ops);
bool scheduleConvNhwcRegions(OpVec &ops);

struct ConvNhwcCacheStats {
    size_t plans = 0;
    size_t weightPrepacks = 0;
    // CPU submissions; CUDA Graph replays do not increment this counter.
    size_t submissions = 0;
    size_t regions = 0;
    size_t regionConvSubmissions = 0;
    size_t regionSiluSubmissions = 0;
    size_t inputTransforms = 0;
    size_t outputTransforms = 0;
};

// Runtime-owned cache for immutable inference weights. The runtime's handle,
// workspace and these scratch buffers must be used serially.
class ConvNhwcCache {
    struct Impl;
    std::unique_ptr<Impl> impl;

  public:
    explicit ConvNhwcCache(int device);
    ~ConvNhwcCache();
    ConvNhwcCache(const ConvNhwcCache &) = delete;
    ConvNhwcCache &operator=(const ConvNhwcCache &) = delete;

    bool run(const Ref<ConvObj> &op, int mode, const CudaRuntimeObj *context,
             size_t *workspaceBytes = nullptr);
    // Returns the consumed prefix length. On an unsupported convolution, the
    // next operator's input is materialized as NCHW before ordinary dispatch.
    size_t runRegion(const OpVec &ops, size_t begin,
                     const CudaRuntimeObj *context);
    ConvNhwcCacheStats stats() const;
};
} // namespace infini
