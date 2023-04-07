#pragma once

#include "data.h"
#include "data_type.h"
#include "op_type.h"
#include <functional>
#include <memory>
#include <unordered_map>
#include <vector>

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

    /// @brief Calculates the size of the tensor in bytes.
    /// @return Memory bytes.
    size_t size() const;

  private:
    /// @brief Constructor is private and only accessible by the factory method.
    Tensor(Vec<size_t> &&, DataType &&, Data &&);
};

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
    /// Every `Operator` must be pushed in topological order.
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

/// @brief A candidate subgraph mutant.
struct Mutant {
    Unigraph graph;
    float score;

    Mutant(Unigraph &&);
    Mutant(Mutant const &) = delete;
    Mutant(Mutant &&);

    Mutant &operator=(Mutant const &) = delete;
    Mutant &operator=(Mutant &&);

    bool operator<(Mutant const &others) const;
    bool operator>(Mutant const &others) const;
};

class Mutation;
class Rating;

/// @brief Partitioned subgraphs.
struct Partition {
    Vec<Vec<Mutant>> graph;
    friend Mutation;

  public:
    using Func = std::function<Vec<Unigraph>(Unigraph &&)>;
    Partition(Unigraph &&, Func const &);
};

/// @brief Generates mutants for every subgraph.
class Mutation {
    Vec<Vec<Mutant>> graph;
    friend Rating;

  public:
    using Func = std::function<Vec<Unigraph>(Unigraph const &)>;
    Mutation(Partition &&, Func const &);
};

/// @brief Rates each subgraph mutant.
class Rating {
    Vec<Vec<Mutant>> graph;

  public:
    using Func = std::function<float(Unigraph const &)>;
    Rating(Mutation &&, Func const &);
};

Vec<Unigraph> split_each(Unigraph &&);
float memory_usage(Unigraph const &);
