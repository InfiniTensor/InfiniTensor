#include "core/common.h"
#include <chrono>
#include <functional>

namespace infini {

double timeit(const std::function<void()> &func) {
    auto start = std::chrono::high_resolution_clock::now();
    func();
    auto end = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double, std::milli>(end - start).count();
}

} // namespace infini