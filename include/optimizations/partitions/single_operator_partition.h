#include "optimizations/partitions/partition.h"

namespace infini {
class SingleOperatorPartition : public Partition {
    Graph run(const GraphObj &, const Transformation &) const override;
};
} // namespace infini
