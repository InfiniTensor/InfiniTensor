#pragma once

#include <cstdint>
#include <cstring>
#include <vector>

/// @brief Stores tensor data。
struct Data {
    /// @brief `cpu_data` is stored in the memory space,
    /// which allows it to be managed using `std::vector<uint8_t>`.
    std::vector<uint8_t> cpu_data;

    // #ifdef USE_CUDA
    //     void *gpu_data;
    // #endif
    // #ifdef USE_BANG
    //     void *mlu_data;
    // #endif

    /// @brief Builds `Data` from `vector` os any type `t`.
    /// @tparam t Data type.
    /// @param data Data `vector`.
    /// @return `Data` object.
    template <class t> static Data cpu(std::vector<t> data) {
        Data ans{std::vector<uint8_t>(sizeof(t) * data.size())};
        memcpy(ans.cpu_data.data(), data.data(), ans.cpu_data.size());
        return ans;
    }
};
