#pragma once

#include "core/runtime.h"

namespace infini {
class MemoryCodegen {
  private:
    std::string generate(Graph graph);

  public:
    MemoryCodegen() {}
    ~MemoryCodegen() {}
    void exportGraph(Graph graph, std::string filename);
    void exportBert_LN(const std::string &filename);
    void exportBert_SM(const std::string &filename);
    void exportBert_GELU(const std::string &filename);
    void exportViT_LN(const std::string &filename);
    void exportViT_SM(const std::string &filename);
    void exportViT_GELU(const std::string &filename);
};
} // namespace infini
