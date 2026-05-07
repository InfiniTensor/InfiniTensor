#pragma once
#include "core/runtime.h"
#include "device.h"
#include <string>
#include <vector>

namespace infini {
namespace test {

inline std::vector<Device::Type> availablePlatforms() {
    std::vector<Device::Type> platforms;
    platforms.push_back(Device::Type::kCpu);
#ifdef WITH_NVIDIA
    platforms.push_back(Device::Type::kNvidia);
#endif
#ifdef WITH_CAMBRICON
    platforms.push_back(Device::Type::kCambricon);
#endif
#ifdef WITH_ASCEND
    platforms.push_back(Device::Type::kAscend);
#endif
#ifdef WITH_METAX
    platforms.push_back(Device::Type::kMetax);
#endif
#ifdef WITH_ILUVATAR
    platforms.push_back(Device::Type::kIluvatar);
#endif
#ifdef WITH_KUNLUN
    platforms.push_back(Device::Type::kKunlun);
#endif
#ifdef WITH_MOORE
    platforms.push_back(Device::Type::kMoore);
#endif
    return platforms;
}

// Factory: create a Runtime for the specified device type
inline Runtime createRuntime(Device::Type type) {
    return make_ref<RuntimeObj>(Device(type));
}

// Human-readable platform name (for test output)
inline std::string platformName(Device::Type type) {
    switch (type) {
    case Device::Type::kCpu:
        return "CPU";
    case Device::Type::kNvidia:
        return "NVIDIA";
    case Device::Type::kCambricon:
        return "Cambricon";
    case Device::Type::kAscend:
        return "Ascend";
    case Device::Type::kKunlun:
        return "KunLun";
    case Device::Type::kMetax:
        return "MetaX";
    case Device::Type::kIluvatar:
        return "Iluvatar";
    case Device::Type::kMoore:
        return "Moore";
    default:
        return "Unknown";
    }
}

} // namespace test
} // namespace infini
