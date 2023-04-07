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

/// @brief A tensor represented by its position in `UniGraph`.
struct TensorPos {
    size_t op, idx;
};

struct Tensor {
    Vec<size_t> shape;
    DataType data_type;
    Data data;
    std::unordered_map<size_t, TensorPos> source;
    std::unordered_map<size_t, Vec<TensorPos>> target;

    /// @brief Tensor memory usage.
    /// @return Memory bytes.
    size_t size() const;
};

struct Operator {
    OpType op_type;
    Vec<Arc<Tensor>> inputs, outputs;
};

struct OpRef {
    size_t graph, op;
};

struct UniGraph {
    size_t id;
    Vec<Operator> operators;

    UniGraph();
    UniGraph(UniGraph const &) = delete;
    UniGraph(UniGraph &&others);
    ~UniGraph();

    UniGraph &operator=(UniGraph const &) = delete;
    UniGraph &operator=(UniGraph &&);

    OpRef push_operator(         // fmt: new line
        OpType op_type,          //
        Vec<Arc<Tensor>> inputs, //
        Vec<Arc<Tensor>> outputs //
    );
};

struct Candidate {
    UniGraph graph;
    float score;

    Candidate(UniGraph &&);
    Candidate(Candidate const &) = delete;
    Candidate(Candidate &&);

    Candidate &operator=(Candidate const &) = delete;
    Candidate &operator=(Candidate &&);

    bool operator<(Candidate const &others) const;
    bool operator>(Candidate const &others) const;
};

class Mutation;
class Rating;

struct Partition {
    Vec<Vec<Candidate>> graph;
    friend Mutation;

  public:
    using Func = std::function<Vec<UniGraph>(UniGraph &&)>;
    Partition(UniGraph &&, Func const &);
};

class Mutation {
    Vec<Vec<Candidate>> graph;
    friend Rating;

  public:
    using Func = std::function<Vec<UniGraph>(UniGraph const &)>;
    Mutation(Partition &&, Func const &);
};

class Rating {
    Vec<Vec<Candidate>> graph;

  public:
    using Func = std::function<float(UniGraph const &)>;
    Rating(Mutation &&, Func const &);
};

Vec<UniGraph> split_each(UniGraph &&);
float memory_usage(UniGraph const &);
