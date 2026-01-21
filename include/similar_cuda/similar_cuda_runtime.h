#pragma once
#include <infiniop/handle.h>
#include <infinirt.h>
#include "core/runtime.h"
#include "utils/exception.h"
#include "core/kernel.h"

namespace infini {
class SimilarRuntimeObj : public RuntimeObj {
    private:
        size_t workspaceSize;
        void *workspace;
        infiniDevice_t device = INFINI_DEVICE_CPU;
        infinirtStream_t stream = nullptr;
    public:
    SimilarRuntimeObj(Device dev) : RuntimeObj(dev, 0) {
        if (dev == Device::ILUVATAR) {
            CHECK_INFINI_ERROR(infinirtSetDevice(INFINI_DEVICE_ILUVATAR, 0));
        } else if (dev == Device::METAX) {
            CHECK_INFINI_ERROR(infinirtSetDevice(INFINI_DEVICE_METAX, 0));
        } else if (dev == Device::MOORE) {
            CHECK_INFINI_ERROR(infinirtSetDevice(INFINI_DEVICE_MOORE, 0));
        }else {
            throw std::runtime_error("Unsupported device");
        }
        CHECK_INFINI_ERROR(infinirtStreamCreate(&stream));
    }

    void run(const Graph &graph, bool tune = false,
             bool profiling = false) const override {
        const auto &kernelRegistry = KernelRegistry::getInstance();
        for (auto &op : graph->getOperators()) {
            auto kernelAttrs = KernelAttrs{handle->device, op->getOpType().underlying()};
            Kernel *kernel = kernelRegistry.getKernel(kernelAttrs);
            kernel->compute(op, this);
        }
    }
    void *alloc(size_t size) override {
        void *ptr;
        CHECK_INFINI_ERROR(infinirtMalloc(&ptr, size));
        return ptr;
    }
    void dealloc(void *ptr) override {
        CHECK_INFINI_ERROR(infinirtFree(ptr));
    }
    void copyBlobFromCPU(void *dst, const void *src,
                                 size_t bytes) const override {
        CHECK_INFINI_ERROR(infinirtMemcpy(dst, src, bytes, INFINIRT_MEMCPY_H2D));
    }
    void copyBlobToCPU(void *dst, const void *src,
                               size_t bytes) const override {
        CHECK_INFINI_ERROR(infinirtMemcpy(dst, src, bytes, INFINIRT_MEMCPY_D2H));
    }
    string toString() const override {
        return "SimilarRuntimeObj";
    }
    void initComm(const string &name, int worldSize, int rank) override { IT_TODO_HALT(); }
    CommunicatorObj &getCommunicator() const override { IT_TODO_HALT(); }
    size_t getWorkspaceSize() const override {
        return workspaceSize;
    }
    void *getCurrentStream() const { return stream; }
    void *getWorkspace(size_t size) const override {
        IT_ASSERT(size < getWorkspaceSize(), "Workspace size is too small");
        return workspace;
    }
    void copyBlobInsideRuntime(void *dst, const void *src,
                               size_t bytes) const override {
        CHECK_INFINI_ERROR(infinirtMemcpy(dst, src, bytes, INFINIRT_MEMCPY_D2D));
    }
};
} // namespace infini
