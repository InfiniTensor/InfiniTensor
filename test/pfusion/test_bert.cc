#include "core/blob.h"
#include "core/graph.h"
#include "core/runtime.h"
#include "operators/element_wise.h"
#include "operators/extend.h"
#include "operators/gather.h"
#include "operators/matmul.h"
#include "operators/reduce_mean.h"
#include "operators/transpose.h"
#include "operators/unary.h"

#include "pfusion/memory_codegen.h"
#include "test.h"

namespace infini {

TEST(Graph, bert_layernorm) {
    MemoryCodegen codegen;
    codegen.exportBert_LN("bert_layernorm.cu");
}

TEST(Graph, bert_softmax) {
    MemoryCodegen codegen;
    codegen.exportBert_SM("bert_softmax.cu");
}

TEST(Graph, bert_gelu) {
    MemoryCodegen codegen;
    codegen.exportBert_GELU("bert_gelu.cu");
}

} // namespace infini
