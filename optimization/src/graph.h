#pragma once

#include "op_type.h"
#include "tensor.h"

namespace optimization {

/// @brief a struct to represent an operator in the computation graph.
///        The ownership of an `Operator` belongs to one `Unigraph`.
struct Operator {
    /// @brief Type of the operator.
    OpType op_type;

    /// @brief Input and output tensors.
    ///        Notice: ownership of the tensors are shared between
    ///        operators that generate and use the same tensor.
    Vec<Arc<Tensor>> inputs, outputs;
};

/// @brief A reference of an `Operator` in a `Unigraph`.
struct OpRef {
    /// @brief `graph` for unique identifier of `Unigraph`.
    ///        `op` for `Operator` index in `Unigraph`.
    size_t graph, op;
};

/// @brief An unpartitioned graph or an unpartitionable minimum graph.
struct Unigraph {
    /// @brief Unique identifier.
    size_t id;
    /// @brief List of operators in the graph with topological order.
    Vec<Operator> operators;

    Unigraph();
    Unigraph(Unigraph const &) = delete;
    Unigraph(Unigraph &&others);
    ~Unigraph();

    Unigraph &operator=(Unigraph const &) = delete;
    Unigraph &operator=(Unigraph &&);

    /// @brief Pushs an `Operator` into graph.
    ///        Every `Operator` must be pushed in topological order.
    /// @param op_type Operator type.
    /// @param inputs Input tensors.
    /// @param outputs Output tensors.
    /// @return An `OpRef`.
    OpRef push_operator(         // fmt: new line
        OpType op_type,          //
        Vec<Arc<Tensor>> inputs, //
        Vec<Arc<Tensor>> outputs //
    );
};

} // namespace optimization
