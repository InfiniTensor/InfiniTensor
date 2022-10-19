#pragma once

#include "micro_kernel.h"

namespace memb {
class MicroKernelElement : MicroKernel {
  private:
    std::string function_name, function_code;

  public:
    MicroKernelElement(std::string function_name_, std::string function_code_)
        : function_name(function_name_), function_code(function_code_) {}
    ~MicroKernelElement() {}
    std::string gen_func();
    std::string gen_kernel(Ptr src, Ptr dst, int m, int n, std::string lda);
};

} // namespace memb