#pragma once

#include "../mutation.h"

namespace optimization::pass {

/// @brief Partition every operator as a `Unigraph`.
struct SingleOperator {};

/// @brief Splits a graph into subgraphs, where each subgraph contains
///        only a single operator.
/// @param arg0 An unpartitioned graph.
/// @return A vector of individual subgraphs.
Vec<std::pair<Unigraph, SingleOperator>> partition(Unigraph &&);

/// @brief Mutates the single operator graph.
/// @param g The subgraph.
/// @param arg1 Never used.
/// @return Mutants.
Vec<Unigraph> mutate(Unigraph const &g, SingleOperator const &);

} // namespace optimization::pass
