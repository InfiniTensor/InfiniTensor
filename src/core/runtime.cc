#include "core/runtime.h"
#include "core/blob.h"
#include "core/kernel.h"
#include "core/perf_engine.h"
#include "utils/data_generator.h"
#include <chrono>
#include <cstdlib>
#include <cstring>

#ifdef WITH_CPU
#include "cpu/runtime_.h"
#endif
#ifdef WITH_NVIDIA
#include "cuda/nvidia/runtime_.h"
#endif
#ifdef WITH_MOORE
#include "cuda/moore/runtime_.h"
#endif
#ifdef WITH_ILUVATAR
#include "cuda/iluvatar/runtime_.h"
#endif
#ifdef WITH_METAX
#include "cuda/metax/runtime_.h"
#endif
#ifdef WITH_CAMBRICON
#include "cambricon/runtime_.h"
#endif
#ifdef WITH_ASCEND
#include "ascend/runtime_.h"
#endif

namespace infini {

namespace {

template <Device::Type DT> struct DeviceTag {
    static constexpr Device::Type value = DT;
};

template <typename F> void dispatchDevice(Device::Type type, F &&func) {
#define DISPATCH_CASE(TYPE)                                                    \
    case Device::Type::TYPE:                                                   \
        std::forward<F>(func)(DeviceTag<Device::Type::TYPE>{});                \
        break;
    switch (type) {
    case Device::Type::kCpu:
        std::forward<F>(func)(DeviceTag<Device::Type::kCpu>{});
        break;
#ifdef WITH_NVIDIA
    case Device::Type::kNvidia:
        std::forward<F>(func)(DeviceTag<Device::Type::kNvidia>{});
        break;
#endif
#ifdef WITH_MOORE
    case Device::Type::kMoore:
        std::forward<F>(func)(DeviceTag<Device::Type::kMoore>{});
        break;
#endif
#ifdef WITH_ILUVATAR
    case Device::Type::kIluvatar:
        std::forward<F>(func)(DeviceTag<Device::Type::kIluvatar>{});
        break;
#endif
#ifdef WITH_METAX
    case Device::Type::kMetax:
        std::forward<F>(func)(DeviceTag<Device::Type::kMetax>{});
        break;
#endif
#ifdef WITH_CAMBRICON
    case Device::Type::kCambricon:
        std::forward<F>(func)(DeviceTag<Device::Type::kCambricon>{});
        break;
#endif
#ifdef WITH_ASCEND
    case Device::Type::kAscend:
        std::forward<F>(func)(DeviceTag<Device::Type::kAscend>{});
        break;
#endif
    default:
        IT_TODO_HALT_MSG("RuntimeObj: unsupported device type " +
                         std::to_string(static_cast<int>(type)));
    }
#undef DISPATCH_CASE
}

} // anonymous namespace

// ---- Constructor ----

RuntimeObj::RuntimeObj(Device device, int deviceId)
    : device(device), deviceId(deviceId) {
    dispatchDevice(device.type(), [&](auto tag) {
        constexpr auto DT = decltype(tag)::value;
        infini::ops::Runtime<DT>::Malloc(&workspace_, workspaceSize_);
    });
}

RuntimeObj::~RuntimeObj() { dealloc(workspace_); }

// ---- alloc / dealloc ----

void *RuntimeObj::alloc(size_t size) {
    void *ptr = nullptr;
    dispatchDevice(device.type(), [&](auto tag) {
        constexpr auto DT = decltype(tag)::value;
        infini::ops::Runtime<DT>::Malloc(&ptr, size);
    });
    return ptr;
}

void RuntimeObj::dealloc(void *ptr) {
    if (!ptr)
        return;
    dispatchDevice(device.type(), [&](auto tag) {
        constexpr auto DT = decltype(tag)::value;
        infini::ops::Runtime<DT>::Free(ptr);
    });
}

// ---- Blob copy ----

void RuntimeObj::copyBlobInsideRuntime(void *dst, const void *src,
                                       size_t bytes) const {
    dispatchDevice(device.type(), [&](auto tag) {
        constexpr auto DT = decltype(tag)::value;
        if constexpr (DT == Device::Type::kCpu) {
            std::memcpy(dst, src, bytes);
        } else {
            // TODO: Uncomment once InfiniOps adds MemcpyDeviceToDevice to
            //       Runtime interface, then remove the IT_TODO_HALT below.
            // infini::ops::Runtime<DT>::Memcpy(
            //     dst, src, bytes,
            //     infini::ops::Runtime<DT>::MemcpyDeviceToDevice);
            IT_TODO_HALT_MSG("copyBlobInsideRuntime: device '" +
                             device.ToString() + "' is not supported");
        }
    });
}

void RuntimeObj::copyBlobFromCPU(void *dst, const void *src,
                                 size_t bytes) const {
    dispatchDevice(device.type(), [&](auto tag) {
        constexpr auto DT = decltype(tag)::value;
        infini::ops::Runtime<DT>::Memcpy(
            dst, src, bytes, infini::ops::Runtime<DT>::MemcpyHostToDevice);
    });
}

void RuntimeObj::copyBlobToCPU(void *dst, const void *src, size_t bytes) const {
    dispatchDevice(device.type(), [&](auto tag) {
        constexpr auto DT = decltype(tag)::value;
        infini::ops::Runtime<DT>::Memcpy(
            dst, src, bytes, infini::ops::Runtime<DT>::MemcpyDeviceToHost);
    });
}

// ---- Workspace / Handle / toString ----

void *RuntimeObj::getWorkspace(size_t size) {
    if (size > workspaceSize_) {
        IT_TODO_HALT_MSG("Workspace size exceeded");
    }
    void *ptr = static_cast<uint8_t *>(workspace_) + workspaceCursor_;
    workspaceCursor_ += size;
    // Align to 256 bytes
    workspaceCursor_ = (workspaceCursor_ + 255) & ~size_t(255);
    return ptr;
}

void RuntimeObj::resetWorkspace() { workspaceCursor_ = 0; }

infini::ops::Handle RuntimeObj::makeHandle() const {
    infini::ops::Handle handle;
    handle.set_workspace(workspace_);
    return handle;
}

string RuntimeObj::toString() const {
    return "Runtime (" + device.ToString() + ")";
}

// ---- Graph execution ----

void RuntimeObj::run(const Graph &graph, bool tune, bool profiling) const {
    if (!tune && profiling)
        IT_TODO_HALT();
    const auto &kernelRegistry = KernelRegistry::getInstance();
    auto &perfEngine = PerfEngine::getInstance();
    // Statistics
    double totalTime = 0;
    std::map<OpType, double> opTime;
    std::map<OpType, int> opCnt;

    for (auto &op : graph->getOperators()) {
        auto kernelAttrs = KernelAttrs{device, op->getOpType().underlying()};
        Kernel *kernel = kernelRegistry.getKernel(kernelAttrs);
        auto perfKey = PerfEngine::Key{kernelAttrs, op->getOpPerfKey()};
        auto perfData = perfEngine.getPerfData(perfKey);

        // If no record and disable tuning, run with the default argument
        if (!perfData && !tune) {
            kernel->compute(op, this);
            continue;
        }

        // TODO: The copy of record should be eliminated
        PerfRecord record;
        // Tune the kernel if there is no record
        if (!perfData) {
            record = kernel->tune(op, this);
            perfEngine.setPerfData(perfKey, record);
        } else
            record = perfData;

        kernel->computeFuncTune(perfKey, op, record, this);
        ComputeFuncPtr funcPtr = kernel->getComputeFunc(perfKey);

        if (!profiling) {
            funcPtr(op, record, this);
            continue;
        } else {
            double t =
                timeit([&]() { funcPtr(op, record, this); }, []() {}, 1, 1);
            op->print();
            printf(" op_time %lf\n", t);
            totalTime += t;
            opTime[op->getOpType()] += t;
            opCnt[op->getOpType()]++;
        }
    }
    if (profiling)
        printProfilingData(totalTime, opTime, opCnt);
}

double RuntimeObj::getPerfTime(const Graph &graph, bool profiling) const {
    const auto &kernelRegistry = KernelRegistry::getInstance();
    auto &perfEngine = PerfEngine::getInstance();
    // Statistics
    double totalTime = 0;
    std::map<OpType, double> opTime;
    std::map<OpType, int> opCnt;

    for (auto &op : graph->getOperators()) {
        auto kernelAttrs = KernelAttrs{device, op->getOpType().underlying()};
        Kernel *kernel = kernelRegistry.getKernel(kernelAttrs);
        auto perfKey = PerfEngine::Key{kernelAttrs, op->getOpPerfKey()};
        auto perfData = perfEngine.getPerfData(perfKey);

        PerfRecord record;
        // Tune the kernel if there is no record
        if (!perfData) {
            TensorVec allocatedTensors;
            for (auto t : op->getInputs())
                if (!t->hasData())
                    allocatedTensors.emplace_back(t);
            for (auto t : op->getOutputs())
                if (!t->hasData())
                    allocatedTensors.emplace_back(t);
            for (auto t : allocatedTensors) {
                t->dataMalloc();
                t->setData(IncrementalGenerator());
            }

            record = kernel->tune(op, this);
            perfEngine.setPerfData(perfKey, record);

            for (auto t : allocatedTensors)
                t->freeData();
        } else
            record = perfData;

        double t = record->time;
        totalTime += t;
        if (profiling) {
            op->print();
            printf(" op_time %lf\n", t);
            opTime[op->getOpType()] += t;
            opCnt[op->getOpType()]++;
        }
    }
    if (profiling)
        printProfilingData(totalTime, opTime, opCnt);
    return totalTime;
}

void RuntimeObj::printProfilingData(double totalTime,
                                    const std::map<OpType, double> &opTime,
                                    const std::map<OpType, int> &opCnt) const {
    printf("%11s %3s %7s %7s %7s\n", "Op", "Cnt", "T_tot", "Percent", "T_mean");
    for (const auto &[type, t] : opTime) {
        printf("%11s %3d %7.3f %7.1f %7.3f\n", type.toString(), opCnt.at(type),
               t, t / totalTime * 100, t / opCnt.at(type));
    }
}

Blob RuntimeObj::allocBlob(size_t size) {
    return make_ref<BlobObj>(shared_from_this(), alloc(size));
}

void RuntimeObj::copyBlob(const TensorObj *dst, const TensorObj *src) const {
    void *dstPtr = dst->getRawDataPtr<void *>();
    void *srcPtr = src->getRawDataPtr<void *>();
    size_t bytes = dst->getBytes();
    auto dstRuntime = dst->getRuntime();
    auto srcRuntime = src->getRuntime();

    if (dstRuntime.get() == srcRuntime.get()) {
        dstRuntime->copyBlobInsideRuntime(dstPtr, srcPtr, bytes);
    } else if (src->getRuntime()->isCpu()) {
        dstRuntime->copyBlobFromCPU(dstPtr, srcPtr, bytes);
    } else if (dst->getRuntime()->isCpu()) {
        srcRuntime->copyBlobToCPU(dstPtr, srcPtr, bytes);
    } else
        IT_TODO_HALT();
}

} // namespace infini
