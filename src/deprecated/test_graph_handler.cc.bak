#include "core/graph_handler.h"
#include "core/runtime.h"
#include <test.h>

namespace infini {

TEST(Handler, matmul) {
    auto runtime = NativeCpuRuntimeObj::getInstance();
    auto handler = make_ref<GraphHandlerObj>(runtime);
    auto i = handler->tensor({1, 2, 3}, DataType::UInt32.getIndex());
    auto w = handler->tensor({1, 3, 4}, DataType::UInt32.getIndex());
    auto o = handler->tensor({1, 2, 4}, DataType::UInt32.getIndex());
    handler->matmul(i, w, o, false, false, nullptr, ActType::None);
}

} // namespace infini
