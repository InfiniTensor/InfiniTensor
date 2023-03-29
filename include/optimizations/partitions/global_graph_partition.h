#include "optimizations/partitions/partition.h"

namespace infini {
class GlobalGraphPartition : public Partition {
    Graph run(const GraphObj &graph,
              const Transformation &transformation) const override;
};
} // namespace infini
