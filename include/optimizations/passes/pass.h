#pragma once

#include "core/graph.h"
#include "optimizations/partitions/partition.h"

namespace infini {
class Pass {
    std::unique_ptr<Partition> p;
    std::unique_ptr<Transformation> tr;
    std::unique_ptr<Rating> rating;

  public:
    Pass(std::unique_ptr<Partition> p, std::unique_ptr<Transformation> tr,
         std::unique_ptr<Rating> rating)
        : p(std::move(p)), tr(std::move(tr)), rating(std::move(rating)) {}

    Graph run(const GraphObj &graph) const {
        return p->run(graph, *tr, *rating);
    }
};
} // namespace infini
