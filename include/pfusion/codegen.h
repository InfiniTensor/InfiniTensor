#pragma once

#include "code.h"

namespace memb {
std::string gen_lowest_basic(int dimn, int dimx, int dimy, int nblocks,
                             int nthreads);
std::string gen_relu(int dimn, int dimx, int dimy, int nblocks, int nthreads);
} // namespace memb