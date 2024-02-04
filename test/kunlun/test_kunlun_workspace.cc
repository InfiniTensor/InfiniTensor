#include "core/runtime.h"
#include "core/workspace.h"
#include "kunlun/kunlun_runtime.h"

#include "test.h"

namespace infini {
TEST(KunlunWorkspace, test) {
    Ref<KUNLUNRuntimeObj> kunlunRuntime = make_ref<KUNLUNRuntimeObj>();
    auto wkspace = kunlunRuntime->getWorkspaceObj();
    KUNLUNPtr space1 = kunlunRuntime->getWorkspace(1024 * 1024 * sizeof(float));
    IT_ASSERT(wkspace->getWorkspaceAlloc() == 1024 * 1024 * sizeof(float));
    KUNLUNPtr space2 = kunlunRuntime->getWorkspace(1024 * 1024 * sizeof(float));
    IT_ASSERT(wkspace->getWorkspaceAlloc() == 1024 * 1024 * sizeof(float) * 2);
    IT_ASSERT((void *)(static_cast<uint8_t *>(space1) +
                       1024 * 1024 * sizeof(float)) == (void *)space2);
    wkspace->resetWorkspace();
    IT_ASSERT(wkspace->getWorkspaceAlloc() == 0);
}
} // namespace infini