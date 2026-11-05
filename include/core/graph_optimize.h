#pragma once

#include "core/graph.h"

namespace infini {

// 对一张图执行所有融合 pass，返回是否做了任何改动
bool optimizeGraph(GraphObj *graph, int maxFusions = -1);

} // namespace infini
