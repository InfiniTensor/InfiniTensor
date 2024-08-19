#pragma once
#include "dbg.h"
#include <cassert>
#include <list>
#include <map>
#include <optional>
#include <set>
#include <string>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace nnet {
using std::dynamic_pointer_cast;
using std::endl;
using std::list;
using std::make_pair;
using std::make_shared;
using std::make_tuple;
using std::map;
using std::max;
using std::min;
using std::nullopt;
using std::optional;
using std::pair;
using std::set;
using std::shared_ptr;
using std::string;
using std::tie;
using std::to_string;
using std::tuple;
using std::unique_ptr;
using std::unordered_map;
template <typename T> using uset = std::unordered_set<T>;
using std::vector;
using std::weak_ptr;

// Aliases
using dtype = float;
using HashType = int;

template <typename T> struct ptr_less {
    bool operator()(const T &lhs, const T &rhs) const { return *lhs < *rhs; }
};

template <typename T> struct ptr_hash {
    size_t operator()(const T &lhs) const {
        return std::hash<decltype(*lhs)>()(*lhs);
    }
};

template <typename T> struct ptr_equal {
    bool operator()(const T &lhs, const T &rhs) const { return *lhs == *rhs; }
};

static inline HashType genhash(HashType a, HashType b) {
    return (a * 10007 + b + 12345) % 1000000007;
}

static inline HashType genhash(string s) {
    HashType ret = 0;
    for (auto c : s)
        ret = genhash(ret, c);
    return ret;
}

#define nnet_unimplemented_halt()                                              \
    { assert(!"Unimplemented"); }

#define nnet_unimplemented_continue()                                          \
    { dbg("Unimplemented"); }

#define nnet_assert(expr, msg) assert(((void)(msg), (expr)))

std::string pointer_to_hex(void *i);
} // namespace nnet
