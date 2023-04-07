#pragma once

#include "data.h"
#include "data_type.h"
#include "op_type.h"
#include <functional>
#include <memory>
#include <numeric>
#include <queue>
#include <unordered_map>
#include <vector>

template <class t> using Vec = std::vector<t>;
template <class t> using Arc = std::shared_ptr<t>;
template <class t>
using MaxQueue = std::priority_queue<t, Vec<t>, std::greater<t>>;

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

    UniGraph() : id(ID++) {}
    UniGraph(UniGraph const &) = delete;
    UniGraph(UniGraph &&others);
    ~UniGraph();

    OpRef push_operator(         // fmt: new line
        OpType op_type,          //
        Vec<Arc<Tensor>> inputs, //
        Vec<Arc<Tensor>> outputs //
    );

  private:
    static size_t ID;
};

struct Candidates {
    void push(UniGraph, float);
    UniGraph pop();

  private:
    struct Candidate {
        size_t index;
        float score;

        bool operator<(Candidate const &others) const {
            return this->score < others.score;
        }

        bool operator>(Candidate const &others) const {
            return this->score > others.score;
        }
    };

    Vec<UniGraph> inner;
    MaxQueue<Candidate> sorter;
};

struct Graph {
    Vec<Candidates> subgraphs;
};
