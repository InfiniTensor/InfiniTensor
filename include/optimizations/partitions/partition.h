#pragma once

#include "core/graph.h"
#include "optimizations/rate/rate.h"
#include "optimizations/transformations/transformation.h"

namespace infini {
class Partition {
  public:
    virtual Graph run(const GraphObj &graph,
                      const Transformation &transformation) const = 0;

  protected:
    std::unique_ptr<Rating> rating;

    /// @brief Rank the subgraph substitutes.
    /// @param subgraph The subgraph to transform.
    /// @param tr Transformation object.
    /// @return Ranked substitutes.
    vector<Graph> rankSubstitutes(const GraphObj &subgraph,
                                  const Transformation &tr) const;
};
} // namespace infini
