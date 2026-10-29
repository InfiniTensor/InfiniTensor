#pragma once
#include "core/graph.h"
#include "core/runtime.h"
#include "execution.h"
namespace infini::infiniops {
struct PreparedGraph {
    WRef<GraphObj> owner;
    uint64_t graphId;
    // Reuse only while the graph generation is unchanged. New mutable
    // operator attributes must notify capture state, as tensor layouts do.
    size_t generation;
    Plan plan;
};
std::shared_ptr<PreparedGraph> prepareGraph(const Graph &graph,
                                            const RuntimeObj *runtime,
                                            MemoryAllocator allocate = {});
} // namespace infini::infiniops
