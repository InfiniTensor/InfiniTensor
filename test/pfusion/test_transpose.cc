#include "pfusion/memory_codegen.h"
#include "test.h"

namespace infini {

TEST(Graph, transpose_0) {
    MemoryCodegen codegen;
    codegen.exportTranspose("transpose_0.cu", {28 * 28, 58, 2}, {0, 2, 1});
}

TEST(Graph, transpose_1) {
    MemoryCodegen codegen;
    codegen.exportTranspose("transpose_1.cu", {14 * 14, 116, 2}, {0, 2, 1});
}

TEST(Graph, transpose_2) {
    MemoryCodegen codegen;
    codegen.exportTranspose("transpose_2.cu", {7 * 7, 232, 2}, {0, 2, 1});
}

} // namespace infini
