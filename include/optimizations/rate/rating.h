#pragma once

#include "core/graph.h"

namespace infini {
/// @brief Rate a `Graph`.
class Rating {
  public:
    /// @brief Cost of a substitute.
    using Cost = float;

    /// @brief Run the `Rating` on the `graph`.
    /// @param graph The graph to rate.
    /// @return The cost of `graph`.
    virtual Cost run(const GraphObj &graph) const = 0;
};
} // namespace infini
