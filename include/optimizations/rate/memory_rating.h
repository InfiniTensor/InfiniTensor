#pragma once

#include "optimizations/rate/rating.h"
#include <numeric>

namespace infini {
/**
 * Rate a `Graph` by its memory usage.
 */
class MemoryRating : public Rating {
  public:
    /**
     * Run the `Rating` on the graph.
     */
    float run(const GraphObj &graph) const override {
        auto tensors = graph.getTensors();
        return static_cast<float>(
            std::accumulate(tensors.begin(), tensors.end(), (size_t)0,
                            [](auto x) { return x.size(); }));
    }
};
} // namespace infini
