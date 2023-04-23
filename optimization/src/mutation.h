#pragma once

#include "graph.h"
#include <functional>

namespace optimization {

/// @brief A candidate subgraph mutant.
struct Mutant {
    /// @brief The mutated subgraph.
    Unigraph graph;

    /// @brief A score representing the quality of the mutant.
    float score;

    Mutant(Unigraph &&g) : graph(std::move(g)) {}
    Mutant(Mutant const &) = delete;
    Mutant(Mutant &&others) : graph(std::move(others.graph)) {}

    Mutant &operator=(Mutant const &) = delete;
    Mutant &operator=(Mutant &&others) {
        if (this != &others)
            this->graph = std::move(others.graph);
        return *this;
    }
};

/// @brief A subgraph partition with `PartitionType`, will be mutated into
///        multiple `Mutant`s.
/// @tparam PartitionType To partition this subgraph.
template <class PartitionType> struct SubGraph {
    Vec<Mutant> mutants;
    PartitionType type;
};

template <class t> Vec<size_t> list_size(Vec<Vec<t>> const &);
template <class PartitionType> class Mutation;
template <class PartitionType> class Rating;

/// @brief Partitioned subgraphs.
template <class PartitionType> struct Partition {
    /// @brief 2D vector of `Mutant` instances for each partitioned subgraph.
    Vec<SubGraph<PartitionType>> parts;

    friend Mutation<PartitionType>;

  public:
    /// @brief A functional object that takes an unpartitioned graph as input
    ///        and returns a vector of partitioned subgraphs.
    using Func =
        std::function<Vec<std::pair<Unigraph, PartitionType>>(Unigraph &&)>;

    /// @brief Constructs a partitioned graph from an unpartitioned graph
    ///        using a partitioning function.
    /// @param g An unpartitioned graph.
    /// @param f A function that takes an unpartitioned graph as input
    /// and returns a vector of partitioned subgraphs.
    Partition(Unigraph &&g, Func const &f) {
        for (auto &[g_, t] : f(std::move(g))) {
            auto &sub = this->parts.emplace_back();
            sub.mutants.emplace_back(std::move(g_));
            sub.type = std::move(t);
        }
    }

    /// @brief Returns mutant vector size.
    /// @return 2D vector size.
    Vec<size_t> size() const { return list_size(parts); }
};

/// @brief Generates mutants for every subgraph.
template <class PartitionType> class Mutation {
    /// @brief 2D vector of `Mutant` instances for each partitioned subgraph.
    Vec<SubGraph<PartitionType>> parts;

    friend Rating<PartitionType>;

  public:
    /// @brief A functional object that takes a subgraph as input
    ///        and returns a vector of mutated graphs.
    using Func =
        std::function<Vec<Unigraph>(Unigraph const &, PartitionType const &)>;

    /// @brief Mutates every subgraph in a partitioned graph.
    /// @param p The partitioned graph to be mutated.
    /// @param f A function that takes a subgraph as input
    /// and returns a vector of mutated graphs.
    Mutation(Partition<PartitionType> &&p, Func const &f)
        : parts(std::move(p.parts)) {
        for (auto &sub : parts)
            for (auto &m : f(sub.mutants.front().graph, sub.type))
                sub.mutants.emplace_back(std::move(m));
    }

    /// @brief Returns mutant vector size.
    /// @return 2D vector size.
    Vec<size_t> size() const { return list_size(parts); }
};

/// @brief Rates each subgraph mutant.
template <class PartitionType> class Rating {
    /// @brief 2D vector of `Mutant` instances for each partitioned subgraph.
    Vec<SubGraph<PartitionType>> parts;

  public:
    /// @brief A functional object that takes a mutated subgraph as input
    ///        and returns its score.
    using Func = std::function<float(Unigraph const &)>;

    /// @brief Rates every mutated subgraph with a `Rating::Func`.
    /// @param m The mutated subgraphs to be rated.
    /// @param f A function that takes a mutated subgraph as input
    ///             and returns its score.
    Rating(Mutation<PartitionType> &&m, Func const &f)
        : parts(std::move(m.parts)) {
        for (auto &sub : parts) {
            auto sum = 0.0f;
            for (auto &c : sub.mutants)
                sum += (c.score = f(c.graph));
            sum = std::abs(sum);
            for (auto &c : sub.mutants)
                c.score /= sum;
            std::sort(
                sub.mutants.begin(), sub.mutants.end(),
                [](auto const &a, auto const &b) { return a.score > b.score; });
        }
    }

    /// @brief Returns mutant vector size.
    /// @return 2D vector size.
    Vec<size_t> size() const { return list_size(parts); }

    /// @brief Builds `Unigraph` from the subgraphs
    /// with specified indices.
    /// @param indices Subgraph indices.
    /// @return Merged `Unigraph`.
    Unigraph build(Vec<size_t> const &indices) const {
        const auto size = indices.size();
        if (size != parts.size())
            throw "indices size wrong";
        Unigraph ans;
        for (size_t i = 0; i < size; ++i)
            for (const auto &op :
                 parts.at(i).mutants.at(indices[i]).graph.operators)
                ans.push_operator(op.op_type, op.inputs, op.outputs);
        return ans;
    }
};

template <class t> Vec<size_t> list_size(Vec<SubGraph<t>> const &list) {
    Vec<size_t> ans(list.size());
    std::transform(list.begin(), list.end(), ans.begin(),
                   [](const auto &e) { return e.mutants.size(); });
    return ans;
}

} // namespace optimization
