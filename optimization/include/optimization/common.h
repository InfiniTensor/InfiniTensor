#pragma once

#include "../../src/mutation.h"
#include "../../src/pass/single_operator.h"
#include <unordered_set>

namespace optimization {

/// @brief Calculates the memory usage of a graph.
/// @param arg0 The graph.
/// @return The reciprocal of the total memory usage of the graph in bytes.
inline float memory_usage(Unigraph const &g) {
    std::unordered_set<size_t> mark;
    uintptr_t memory;
    for (const auto &op : g.operators)
        for (const auto &t : op.outputs)
            if (mark.insert(reinterpret_cast<uintptr_t>(t.get())).second)
                memory += t->size();
    return 1e6f / static_cast<float>(memory);
}

} // namespace optimization
