#pragma once

#include "optimizations/rate/rating.h"

namespace infini {
/**
 * Rate a `Graph` by its memory usage.
 */
class TimeRating : public Rating {
  public:
    /**
     * Run the `Rating` on the graph.
     */
    float run(const GraphObj &graph) const override;
};
} // namespace infini
