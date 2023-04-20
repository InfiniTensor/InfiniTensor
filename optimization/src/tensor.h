#pragma once

#include "data.h"
#include "data_type.h"
#include <memory>
#include <unordered_map>
#include <vector>

namespace optimization {

/// @brief Defines a template alias for `std::vector`.
template <class t> using Vec = std::vector<t>;

/// @brief Defines a template alias for std::shared_ptr
template <class t> using Arc = std::shared_ptr<t>;

/// @brief A tensor represented by its position in `Unigraph`.
struct TensorPos {
    /// @brief `op` for `Operator` index in `Unigraph`.
    ///        `idx` for index in `Operator` inputs or outputs.
    size_t op, idx;
};

/// @brief A struct to represent a tensor in the computation graph.
///        The ownership of a `Tensor` belongs to all the operators
///        that generate it or it passed to.
struct Tensor {
    /// @brief Tensor shape.
    Vec<size_t> shape;

    /// @brief Element data type.
    DataType data_type;

    /// @brief Data of tensor.
    Data data;

    /// @brief Operators in different `Unigraph` that generate this tensor.
    std::unordered_map<size_t, TensorPos> source;

    /// @brief Operators in different `Unigraph` that take this tensor as input.
    std::unordered_map<size_t, Vec<TensorPos>> target;

    /// @brief A static factory method to create a `shared_ptr<Tensor>`.
    /// @param shape Tensor shape.
    /// @param data_type Element data type.
    /// @param data Data.
    /// @return A `shared_ptr<Tensor>`.
    static Arc<Tensor> share(Vec<size_t> shape, DataType data_type, Data data);

    /// @brief A static factory method to create a `shared_ptr<Tensor>` with
    /// single data.
    /// @tparam t Data type.
    /// @param val Data value.
    /// @return A `shared_ptr<Tensor>`.
    template <class t> static Arc<Tensor> share_single(t val) {
        return Tensor::share({1}, ty<t>(), Data::cpu<t>({val}));
    }

    /// @brief A static factory method to create a `shared_ptr<Tensor>` with
    /// 1D data.
    /// @tparam t Data type.
    /// @param val Data value.
    /// @return A `shared_ptr<Tensor>`.
    template <class t> static Arc<Tensor> share_vec(Vec<t> val) {
        return Tensor::share({val.size()}, ty<t>(),
                             Data::cpu<t>(std::move(val)));
    }

    /// @brief Calculates the size of the tensor in bytes.
    /// @return Memory usage in bytes.
    size_t size() const;

  private:
    /// @brief Constructor is private and only accessible by the factory method.
    Tensor(Vec<size_t> &&, DataType &&, Data &&);
};

} // namespace optimization
