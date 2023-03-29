#pragma once

#include "core/graph.h"

namespace infini {
/**
 * Rate a `Graph`.
 */
class Rating {
  public:
    /**
     * Run the `Rating` on the graph.
     */
    virtual float run(const GraphObj &graph) = 0;
};
} // namespace infini
