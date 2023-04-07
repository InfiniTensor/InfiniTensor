#pragma once

#include "data.h"
#include "data_type.h"
#include "op_type.h"
#include <functional>
#include <memory>
#include <unordered_map>
#include <vector>

template <class t> using Vec = std::vector<t>;
template <class t> using Arc = std::shared_ptr<t>;

/// @brief A tensor represented by its position in `Unigraph`.
struct TensorPos {
    size_t op, idx;
};

/// @brief The ownership of a `Tensor` belongs to all the operators
/// that generate it or it passed to.
struct Tensor {
    Vec<size_t> shape;
    DataType data_type;
    Data data;
    std::unordered_map<size_t, TensorPos> source;
    std::unordered_map<size_t, Vec<TensorPos>> target;

    /// @brief Builds a `Tensor` in `std::shared_ptr`.
    /// @param shape Tensor shape.
    /// @param data_type Element data type.
    /// @param data Data.
    /// @return A `shared_ptr<Tensor>`.
    static Arc<Tensor> share(Vec<size_t> shape, DataType data_type, Data data);

    /// @brief Tensor memory usage.
    /// @return Memory bytes.
    size_t size() const;

  private:
    Tensor(Vec<size_t> &&, DataType &&, Data &&);
};

/// @brief The ownership of a `Operator` belongs to one `Unigraph`.
struct Operator {
    OpType op_type;
    Vec<Arc<Tensor>> inputs, outputs;
};

/// @brief A reference of an `Operator` in a `Unigraph`.
struct OpRef {
    size_t graph, op;
};

/// @brief An unpartitioned graph or an unpartitionable minimum graph.
struct Unigraph {
    size_t id;
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
