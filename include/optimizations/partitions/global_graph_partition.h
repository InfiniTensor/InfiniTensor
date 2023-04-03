#include "optimizations/partitions/partition.h"

namespace infini {
class GlobalGraphPartition : public Partition {
    Graph run(const GraphObj &graph, const Transformation &tr,
              const Rating &rating) const override {
        return rankCandidates(graph, tr, rating).top().graph;
    }
};
} // namespace infini
