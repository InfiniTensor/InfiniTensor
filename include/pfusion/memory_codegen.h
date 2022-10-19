#pragma once

#include "core/runtime.h"

namespace infini {
class MemoryCodegen {
  private:
    std::string generate(Graph graph);

  public:
    MemoryCodegen() {}
    ~MemoryCodegen() {}
    void export_code(Graph graph, std::string filename);
};
} // namespace infini
