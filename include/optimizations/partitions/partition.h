#pragma once

#include "core/graph.h"
#include "optimizations/rate/rating.h"
#include "optimizations/transformations/transformation.h"

namespace infini {
class Partition {
  public:
    virtual Graph run(const GraphObj &, const Transformation &,
                      const Rating &) const = 0;

  protected:
    /// @brief Rank the subgraph substitutes.
    /// @param subgraph The subgraph to transform.
    /// @param tr Transformation object.
    /// @return Ranked substitutes.
    vector<Graph> rankSubstitutes(const GraphObj &subgraph,
                                  const Transformation &tr,
                                  const Rating &rating) const;
};
} // namespace infini
