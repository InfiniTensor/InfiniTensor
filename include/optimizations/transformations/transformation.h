#pragma once

#include "core/common.h"
#include "core/graph.h"
#include "core/runtime.h"

namespace infini {
class Transformation {
  public:
    virtual vector<Graph> run(const GraphObj &graph) const {
        return {make_ref<GraphObj>(graph)};
    };
};
} // namespace infini
