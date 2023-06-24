#pragma once
#include "core/graph.h"
#include "core/runtime.h"

namespace infini {

Graph convertNCHWtoNHWCModel(Runtime runtime, Graph inG);

} // namespace infini
