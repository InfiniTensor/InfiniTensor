#pragma once

#include "core/graph.h"
#include "optimizations/partitions/partition.h"

namespace infini {
class Pass {
  public:
    virtual Graph run(const Graph graph) = 0;
};
} // namespace infini
