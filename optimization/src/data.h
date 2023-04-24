#pragma once

#include <cstdint>
#include <cstring>
#include <utility>
#include <vector>

namespace optimization {

/// @brief Stores tensor data。
class Data {
    /// @brief `cpu_data` is stored in the memory space,
    /// which allows it to be managed using `std::vector<uint8_t>`.
    uint8_t *cpu_data;

    // #ifdef USE_CUDA
    //     void *gpu_data;
    // #endif

    // #ifdef USE_BANG
    //     void *mlu_data;
    // #endif

    Data(uint8_t *ptr) : cpu_data(ptr) {}

  public:
    Data() : cpu_data(nullptr) {}
    Data(size_t size) : cpu_data(new uint8_t[size]) {}
    template <class t> Data(t begin, t end) : cpu_data(nullptr) {
        size_t c = sizeof(decltype(*begin)) * static_cast<size_t>(end - begin);
        cpu_data = new uint8_t[c];
        std::copy(begin, end, cpu_data);
    }
    Data(Data const &) = delete;
    Data(Data &&others) noexcept
        : cpu_data(std::exchange(others.cpu_data, nullptr)) {}
    ~Data() noexcept { delete[] cpu_data; }

    Data &operator=(Data const &) = delete;
    Data &operator=(Data &&others) noexcept {
        if (this != &others)
            delete[] std::exchange(cpu_data,
                                   std::exchange(others.cpu_data, nullptr));

        return *this;
    }

    /// @brief Builds `Data` from `vector` os any type `t`.
    /// @tparam t Data type.
    /// @param data Data `vector`.
    /// @return `Data` object.
    template <class t> static Data cpu(std::vector<t> const &data) {
        auto const len = data.size();
        auto const size = sizeof(t[len]);
        Data ans;
        memcpy(ans.cpu_data, data.data(), size);
        return ans;
    }

    /// @brief Gets data ptr.
    /// @tparam t Data type.
    /// @return Data ptr.
    template <class t> t *as_ptr() const {
        return reinterpret_cast<t *>(cpu_data);
    }

    /// @brief Copies data to a `Vec`.
    /// @tparam t Data type.
    /// @param len Count of data.
    /// @return The data `Vec`.
    template <class t> std::vector<t> to_vec(size_t len) const {
        std::vector<t> ans(len);
        memcpy(cpu_data, ans.data(), sizeof(t[len]));
        return ans;
    }
};

} // namespace optimization
