#pragma once

#include "data.h"
#include "op_type.h"
#include "tensor.h"
#include <functional>

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

/// @brief A candidate subgraph mutant.
struct Mutant {
    /// @brief The mutated subgraph.
    Unigraph graph;

    /// @brief A score representing the quality of the mutant.
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
    /// @brief 2D vector of `Mutant` instances for each partitioned subgraph.
    Vec<Vec<Mutant>> mutant;

    friend Mutation;

  public:
    /// @brief A functional object that takes an unpartitioned graph as input
    ///        and returns a vector of partitioned subgraphs.
    using Func = std::function<Vec<Unigraph>(Unigraph &&)>;

    /// @brief Constructs a partitioned graph from an unpartitioned graph
    ///        using a partitioning function.
    /// @param arg0 An unpartitioned graph.
    /// @param arg1 A function that takes an unpartitioned graph as input
    /// and returns a vector of partitioned subgraphs.
    Partition(Unigraph &&, Func const &);

    /// @brief Returns mutant vector size.
    /// @return 2D vector size.
    Vec<size_t> size() const;
};

/// @brief Generates mutants for every subgraph.
class Mutation {
    /// @brief 2D vector of `Mutant` instances for each partitioned subgraph.
    Vec<Vec<Mutant>> mutant;

    friend Rating;

  public:
    /// @brief A functional object that takes a subgraph as input
    ///        and returns a vector of mutated graphs.
    using Func = std::function<Vec<Unigraph>(Unigraph const &)>;

    /// @brief Mutates every subgraph in a partitioned graph.
    /// @param arg0 The partitioned graph to be mutated.
    /// @param arg1 A function that takes a subgraph as input
    /// and returns a vector of mutated graphs.
    Mutation(Partition &&, Func const &);

    /// @brief Returns mutant vector size.
    /// @return 2D vector size.
    Vec<size_t> size() const;
};

/// @brief Rates each subgraph mutant.
class Rating {
    /// @brief 2D vector of `Mutant` instances for each partitioned subgraph.
    Vec<Vec<Mutant>> mutant;

  public:
    /// @brief A functional object that takes a mutated subgraph as input
    ///        and returns its score.
    using Func = std::function<float(Unigraph const &)>;

    /// @brief Rates every mutated subgraph with a `Rating::Func`.
    /// @param arg0 The mutated subgraphs to be rated.
    /// @param arg1 A function that takes a mutated subgraph as input
    ///             and returns its score.
    Rating(Mutation &&, Func const &);

    /// @brief Returns mutant vector size.
    /// @return 2D vector size.
    Vec<size_t> size() const;

    /// @brief Builds `Unigraph` from the subgraphs
    /// with specified indices.
    /// @param arg0 Subgraph indices.
    /// @return Merged `Unigraph`.
    Unigraph build(Vec<size_t> const &) const;
};

/// @brief Splits a graph into subgraphs, where each subgraph contains
///        only one operator.
/// @param arg0 An unpartitioned graph.
/// @return A vector of individual subgraphs.
Vec<Unigraph> split_each(Unigraph &&);

/// @brief Calculates the memory usage of a graph.
/// @param arg0 The graph.
/// @return The reciprocal of the total memory usage of the graph in bytes.
float memory_usage(Unigraph const &);
