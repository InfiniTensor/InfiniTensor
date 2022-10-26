#pragma once

#include "core/runtime.h"

namespace infini {
class MemoryCodegen {
  private:
    std::string generate(Graph graph);

  public:
    MemoryCodegen() {}
    ~MemoryCodegen() {}
    void exportCode(Graph graph, std::string filename);
};
} // namespace infini
