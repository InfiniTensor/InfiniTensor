#include "core/common.h"
#include <chrono>
#include <functional>

namespace infini {

double timeit(const std::function<void()> &func,
              const std::function<void(void)> &sync, int warmupRounds,
              int timingRounds) {
    for (int i = 0; i < warmupRounds; ++i)
        func();
    if (sync)
        sync();
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < timingRounds; ++i)
        func();
    if (sync)
        sync();
    auto end = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double, std::milli>(end - start).count() /
           timingRounds;
}

} // namespace infini
