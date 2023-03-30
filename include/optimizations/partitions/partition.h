#pragma once

#include "core/graph.h"
#include "optimizations/rate/rating.h"
#include "optimizations/transformations/transformation.h"
#include <queue>

namespace infini {
class Partition {
  public:
    virtual Graph run(const GraphObj &, const Transformation &,
                      const Rating &) const = 0;
    struct Candidate {
        Graph graph;
        Rating::Cost cost;

        bool operator<(Candidate others) const { return cost < others.cost; }
        bool operator>(Candidate others) const { return cost > others.cost; }
    };

  protected:
    using CandidateQueue = std::priority_queue<Candidate, vector<Candidate>,
                                               std::greater<Candidate>>;

    /// @brief Rank the subgraph candidates.
    /// @param subgraph The subgraph to transform.
    /// @param tr Transformation object.
    /// @return Ranked candidates.
    CandidateQueue rankCandidates(const GraphObj &subgraph,
                                  const Transformation &tr,
                                  const Rating &rating) const;
};
} // namespace infini
