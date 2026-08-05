#include "core/tensor_base.h"
#include "core/blob.h"
#include "core/graph.h"
#include "core/runtime.h"
#include <algorithm>
namespace infini {

TensorBaseObj::TensorBaseObj(int dim, DataType dtype, Runtime runtime)
    : dim(dim), dtype(dtype), runtime(runtime) {}

void TensorBaseObj::dataMalloc(const Blob &blob) {
    IT_ASSERT(data == nullptr);
    data = blob;
    notifyCaptureState(true);
}

void TensorBaseObj::freeData() {
    if (!data)
        return;
    data.reset();
    notifyCaptureState(true);
}

void TensorBaseObj::registerCaptureState(
    const Ref<GraphCaptureStateObj> &state) {
    const auto stateId = state->getId();
    for (auto it = captureStates.begin(); it != captureStates.end();) {
        if (auto current = it->lock()) {
            if (current->getId() == stateId)
                return;
            ++it;
        } else {
            it = captureStates.erase(it);
        }
    }
    captureStates.emplace_back(state);
}

void TensorBaseObj::unregisterCaptureState(uint64_t stateId) {
    captureStates.erase(
        std::remove_if(captureStates.begin(), captureStates.end(),
                       [stateId](const auto &state) {
                           auto current = state.lock();
                           return !current || current->getId() == stateId;
                       }),
        captureStates.end());
}

void TensorBaseObj::notifyCaptureState(bool storageChanged) {
    for (auto it = captureStates.begin(); it != captureStates.end();) {
        if (auto state = it->lock()) {
            state->markChanged(storageChanged);
            ++it;
        } else {
            it = captureStates.erase(it);
        }
    }
}

}; // namespace infini
