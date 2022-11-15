#include "pfusion/memory_codegen.h"
#include "test.h"

namespace infini {

TEST(Graph, bias_0) {
    MemoryCodegen codegen;
    codegen.exportBias("bias_0.cu", std::vector<size_t>({28 * 28, 24}));
}

TEST(Graph, bias_1) {
    MemoryCodegen codegen;
    codegen.exportBias("bias_1.cu", std::vector<size_t>({28 * 28, 58}));
}

TEST(Graph, bias_2) {
    MemoryCodegen codegen;
    codegen.exportBias("bias_2.cu", std::vector<size_t>({14 * 14, 116}));
}

TEST(Graph, bias_3) {
    MemoryCodegen codegen;
    codegen.exportBias("bias_3.cu", std::vector<size_t>({7 * 7, 232}));
}

} // namespace infini
