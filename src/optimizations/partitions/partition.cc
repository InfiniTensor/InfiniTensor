#include "optimizations/partitions/partition.h"
#include <algorithm>

namespace infini {
Partition::CandidateQueue
Partition::rankCandidates(const GraphObj &subgraph, const Transformation &tr,
                          const Rating &rating) const {
    auto substitutes = tr.run(subgraph);
    CandidateQueue ans;
    while (!substitutes.empty()) {
        auto g = std::move(substitutes.back());
        auto cost = rating.run(*g);
        ans.push({std::move(g), cost});
        substitutes.pop_back();
    }
    return ans;
}

} // namespace infini
