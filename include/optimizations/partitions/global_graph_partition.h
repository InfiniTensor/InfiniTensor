#include "optimizations/partitions/partition.h"

namespace infini {
class GlobalGraphPartition : public Partition {
    Graph run(const GraphObj &graph, const Transformation &tr) const override {
        return rankSubstitutes(graph, tr)[0];
    }
};
} // namespace infini
