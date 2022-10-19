#pragma once

#include "micro_kernel.h"

namespace memb {
class MicroKernelTranspose : MicroKernel {
  public:
    MicroKernelTranspose() {}
    ~MicroKernelTranspose() {}
    std::string generate(Ptr src, Ptr dst, int m, std::string lda, int n,
                         std::string ldb);
};

} // namespace memb