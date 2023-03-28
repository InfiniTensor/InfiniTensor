#include "optimizations/partitions/partition.h"

namespace infini {
class SingleOperatorPartition : public Partition {
    Graph run(const Graph graph, Ref<Transformation> transformation) override;
};
} // namespace infini