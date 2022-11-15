#pragma once

#include "core/runtime.h"

namespace infini {
class MemoryCodegen {
  private:
    std::string generateGraph(Graph graph);
    std::string generateBias(const std::vector<size_t> &shape);
    std::string generateTranspose(const std::vector<size_t> &shape,
                                  const std::vector<size_t> &perm);

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
    void exportBias(const std::string &filename,
                    const std::vector<size_t> &shape);
    void exportTranspose(const std::string &filename,
                         const std::vector<size_t> &shape,
                         const std::vector<size_t> &perm);
};
} // namespace infini
