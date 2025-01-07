#pragma once
#include "utils/exception.h"
#include <cassert>
#include <cstdint>
#include <functional>
#include <iostream>
#include <list>
#include <map>
#include <optional>
#include <set>
#include <sstream>
#include <string>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <variant>
#include <vector>

namespace infini {
using std::list;
using std::map;
using std::optional;
using std::pair;
using std::set;
using std::string;
using std::tie;
using std::to_string;
using std::tuple;
using std::unordered_map;
using std::vector;

// Aliases
using dtype = float;
using HashType = uint64_t; // compatible with std::hash

// Metaprogramming utilities
#define _CAT(A, B) A##B
#define _SELECT(NAME, NUM) _CAT(NAME##_, NUM)
#define _GET_COUNT(_1, _2, _3, _4, _5, _6, _7, _8, _9, _10, COUNT, ...) COUNT
#define _VA_SIZE(...) _GET_COUNT(__VA_ARGS__, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1)
#define _VA_SELECT(NAME, ...) _SELECT(NAME, _VA_SIZE(__VA_ARGS__))(__VA_ARGS__)

// Assert: conditions should have no side effect
#define _IT_ASSERT_2(condition, info)                                          \
    static_cast<bool>(condition)                                               \
        ? void(0)                                                              \
        : throw ::infini::Exception(                                           \
              std::string("[") + __FILE__ + ":" + std::to_string(__LINE__) +   \
              "] Assertion failed (" + #condition + "): " + info)
#define _IT_ASSERT_1(condition) _IT_ASSERT_2(condition, "")
#define IT_ASSERT(...) _VA_SELECT(_IT_ASSERT, __VA_ARGS__)

#define IT_TODO_HALT() _IT_ASSERT_2(false, "Unimplemented")
#define IT_TODO_HALT_MSG(msg) _IT_ASSERT_2(false, msg)
#define IT_ASSERT_TODO(condition) _IT_ASSERT_2(condition, "Unimplemented")
#define IT_TODO_SKIP() puts("Unimplemented " __FILE__ ":" __LINE__)

// Other utilities

// std::to_underlying is avaiable since C++23
template <typename T> auto enum_to_underlying(T e) {
    return static_cast<std::underlying_type_t<T>>(e);
}

template <typename T> std::string vecToString(const std::vector<T> &vec) {
    std::stringstream ss;
    ss << "[";
    for (size_t i = 0; i < vec.size(); ++i) {
        ss << vec.at(i);
        if (i < vec.size() - 1) {
            ss << ",";
        }
    }
    ss << "]";
    return ss.str();
}

template <typename T> std::string vecToString(const T *st, size_t length) {
    std::stringstream ss;
    ss << "[";
    size_t i = 0;
    for (i = 0; i < length; i++) {
        ss << *(st + i);
        if (i < length - 1) {
            ss << ",";
        }
    }
    ss << "]";
    return ss.str();
}

double timeit(
    const std::function<void()> &func,
    const std::function<void(void)> &sync = []() {}, int warmupRounds = 10,
    int timingRounds = 10);

std::vector<int64_t> castTo64(std::vector<int> const &v32);

} // namespace infini
