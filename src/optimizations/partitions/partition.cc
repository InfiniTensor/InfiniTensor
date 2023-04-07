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

namespace x {

struct Operator;

/// @brief 未分的完整图或不可再分的最小子图。
using UniGraph = std::vector<Operator>;
struct Candidate {
    /// @brief 候选子图。
    UniGraph graph;
    /// @brief 子图评分。
    float score;
};
/// @brief 一组连接到相同张量、平行的图。
using Candidates = std::priority_queue<Candidate>;
/// @brief 由多个通过张量相连的子图组合成的完整的图。
using Graph = std::vector<Candidates>;

}; // namespace x
