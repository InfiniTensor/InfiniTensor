#pragma once
#include "common.h"
#include "core/ref.h"
#include <functional> // hash
#include <memory>
#include <type_traits>

namespace nnet {

template <typename T> using Ref = infini::Ref<T>;

template <typename T, typename... Params> Ref<T> make_ref(Params &&...params) {
    return infini::make_ref<T>(std::forward<Params>(params)...);
}

template <class T, class U,
          typename std::enable_if_t<std::is_base_of_v<U, T>> * = nullptr>
Ref<T> as(const Ref<U> &ref) {
    return infini::as<T>(ref);
}

// Comparator for Ref
template <typename T> struct is_ref : std::false_type {};
template <typename T> struct is_ref<Ref<T>> : std::true_type {};

template <class Tuple, std::size_t index = 0, bool address_based>
typename std::enable_if_t<not is_ref<std::tuple_element_t<index, Tuple>>::value,
                          bool>
__ref_less(const Tuple &lhs, const Tuple &rhs) {
    if constexpr (index >=
                  std::tuple_size<std::remove_reference_t<Tuple>>::value - 1)
        return std::get<index>(lhs) < std::get<index>(rhs);
    else {
        if (std::get<index>(lhs) != std::get<index>(rhs))
            return std::get<index>(lhs) < std::get<index>(rhs);
        else
            return __ref_less<Tuple, index + 1, address_based>(lhs, rhs);
    }
}

template <class Tuple, std::size_t index = 0, bool address_based>
typename std::enable_if_t<is_ref<std::tuple_element_t<index, Tuple>>::value and
                              not address_based,
                          bool>
__ref_less(const Tuple &lhs, const Tuple &rhs) {
    if constexpr (index >=
                  std::tuple_size<std::remove_reference_t<Tuple>>::value - 1)
        return std::get<index>(lhs)->less(std::get<index>(rhs));
    else {
        if (std::get<index>(lhs)->neq(std::get<index>(rhs)))
            return std::get<index>(lhs)->less(std::get<index>(rhs));
        else
            return __ref_less<Tuple, index + 1, address_based>(lhs, rhs);
    }
}

template <class Tuple, std::size_t index = 0, bool address_based>
typename std::enable_if_t<
    is_ref<std::tuple_element_t<index, Tuple>>::value and address_based, bool>
__ref_less(const Tuple &lhs, const Tuple &rhs) {
    if constexpr (index >=
                  std::tuple_size<std::remove_reference_t<Tuple>>::value - 1)
        return std::get<index>(lhs).get() < std::get<index>(rhs).get();
    else {
        if (std::get<index>(lhs).get() != std::get<index>(rhs).get())
            return std::get<index>(lhs).get() < std::get<index>(rhs).get();
        else
            return __ref_less<Tuple, index + 1, address_based>(lhs, rhs);
    }
}

template <class Tuple> bool ref_addr_less(const Tuple &lhs, const Tuple &rhs) {
    return __ref_less<Tuple, 0, true>(lhs, rhs);
}

template <class Tuple> bool ref_value_less(const Tuple &lhs, const Tuple &rhs) {
    return __ref_less<Tuple, 0, false>(lhs, rhs);
}

template <class Tuple> class RefAddrLess {
  public:
    bool operator()(const Tuple &a, const Tuple &b) const {
        return ref_addr_less(a, b);
    }
};

template <class Tuple> class RefValueLess {
  public:
    bool operator()(const Tuple &a, const Tuple &b) const {
        return ref_value_less(a, b);
    }
};

// make_ref_from_tuple
template <typename _Tp, typename _Tuple, size_t... _Idx>
constexpr Ref<_Tp> make_ref_from_tuple_impl(_Tuple &&__t,
                                            std::index_sequence<_Idx...>) {
    return make_ref<_Tp>(std::get<_Idx>(std::forward<_Tuple>(__t))...);
}

template <typename _Tp, typename _Tuple>
constexpr Ref<_Tp> make_ref_from_tuple(_Tuple &&__t) {
    return make_ref_from_tuple_impl<_Tp>(
        std::forward<_Tuple>(__t),
        std::make_index_sequence<std::tuple_size_v<std::decay_t<_Tuple>>>{});
}

} // namespace nnet

// namespace std {

// template <class T> struct hash<ir::Ref<T>> {
//     hash<T *> hash_;
//     size_t operator()(const ir::Ref<T> &ref) const { return hash_(ref.get());
//     }
// };

// } // namespace nnet
