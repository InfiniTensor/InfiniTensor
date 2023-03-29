#include "optimizations/partitions/partition.h"

namespace infini {
class SingleOperatorPartition : public Partition {
    Graph run(const GraphObj &graph,
              const Transformation &transformation) const override;
};
} // namespace infini
