#pragma once
#include "common.h"
#include <functional> // hash
#include <memory>
#include <type_traits>

namespace nnet {

template <typename T> struct is_ref;

/**
 * Ref-counting pointer
 *
 * This class is thread-safe (For developers: concurrent accesses through
 * different `std::shared_ptr`s to the same object is already thread-safe, while
 * modifying the same `std::shared_ptr` is not. We never modify a `Ref`, so no
 * locks are needed. See https://en.cppreference.com/w/cpp/memory/shared_ptr)
 */
template <class T> class Ref {
    static_assert(is_ref<T>::value == false, "Ref should not be nested");

    template <class U> friend class Ref;

    std::shared_ptr<T> ptr_;

  private:
  public:
    typedef T Object;

    Ref() = default;
    // Ref(std::nullptr_t) : Ref() {}
    constexpr Ref(nullptr_t) noexcept : Ref() {}
    Ref(const Ref &) = default;
    Ref(Ref &&) = default;
    Ref(std::shared_ptr<T> &&ptr) : ptr_(std::move(ptr)) {}
    // Ref(const std::shared_ptr<T> &ptr) : ptr_(ptr) {}

    // /// NO NOT USE THIS CONSTRUCTOR IN PUBLIC
    // /// It is public because Pybind11 needs it
    // Ref(T *ptr) : ptr_(ptr) {}

    /**
     * Shared with any compatible references
     */
    template <class U,
              typename std::enable_if_t<std::is_base_of_v<T, U>> * = nullptr>
    Ref(const Ref<U> &other) : ptr_(std::static_pointer_cast<T>(other.ptr_)) {}

    template <class U,
              typename std::enable_if_t<std::is_base_of_v<T, U>> * = nullptr>
    Ref &operator=(const Ref<U> &other) {
        ptr_ = std::static_pointer_cast<T>(other.ptr_);
        return *this;
    }

    Ref &operator=(const Ref &) = default;
    Ref &operator=(Ref &&) = default;

    template <class U> Ref<U> as() const {
        Ref<U> ret;
        ret.ptr_ = std::dynamic_pointer_cast<U>(ptr_);
        return ret;
    }

    bool isValid() const { return ptr_ != nullptr; }

    T &operator*() const {
        nnet_assert(isValid(), "Empty pointer.");
        return *ptr_;
    }

    T *operator->() const {
        nnet_assert(isValid(), "Empty pointer.");
        return ptr_.get();
    }

    T *get() const {
        nnet_assert(isValid(), "Empty pointer.");
        return ptr_.get();
    }

    friend inline bool operator==(const Ref &lhs, nullptr_t) {
        return !lhs.isValid();
    }
    friend inline bool operator!=(const Ref &lhs, nullptr_t) {
        return !(lhs == nullptr);
    }
    explicit operator bool() const { return ptr_ != nullptr; }
    bool operator!() { return ptr_ == nullptr; }

    void swap(Ref &__b) noexcept { ptr_.swap(__b.ptr_); }
};

template <class T, class U,
          typename std::enable_if_t<std::is_base_of_v<U, T>> * = nullptr>
Ref<T> as(const Ref<U> &ref) {
    return ref.template as<T>();
}

template <typename T, typename... Params> Ref<T> make_ref(Params &&...params) {
    return Ref(make_shared<T>(std::forward<Params>(params)...));
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