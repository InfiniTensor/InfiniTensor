#include "optimizations/partitions/partition.h"
#include <algorithm>

namespace infini {
vector<Graph> Partition::rankSubstitutes(const GraphObj &subgraph,
                                         const Transformation &tr,
                                         const Rating &rating) const {
    struct RatedGraph {
        size_t index;
        float rate;

        bool operator<(RatedGraph others) const { return rate > others.rate; };
    };
    auto substitutes = tr.run(subgraph);
    auto size = substitutes.size();
    vector<RatedGraph> rank(size);
    vector<Graph> ans(size);
    for (size_t i = 0; i < size; ++i)
        rank[i] = {i, rating.run(*substitutes[i])};
    std::sort(rank.begin(), rank.end());
    std::transform(rank.begin(), rank.end(), ans.begin(),
                   [&substitutes](auto x) { return substitutes[x.index]; });
    return ans;
}

} // namespace infini
