#include "core/graph_handler.h"
#include "core/runtime.h"
#include <test.h>

namespace infini {

TEST(Handler, matmul) {
    auto runtime = CpuRuntimeObj::getInstance();
    auto handler = make_ref<GraphHandlerObj>(runtime);
    auto i = handler->tensor({1, 2, 3}, OnnxDType::UINT32);
    auto w = handler->tensor({1, 3, 4}, OnnxDType::UINT32);
    auto o = handler->tensor({1, 2, 4}, OnnxDType::UINT32);
    handler->matmul(i, w, o, false, false, nullptr, ActType::None);
}

} // namespace infini
