#pragma once

#include "pfusion/common.h"

namespace memb {
std::vector<std::shared_ptr<MetaOp>> instantiateAbs(std::vector<int> shape);
std::vector<std::shared_ptr<MetaOp>> instantiateRelu(std::vector<int> shape);
} // namespace memb
