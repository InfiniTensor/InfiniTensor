#pragma once

#include "../mutation.h"

namespace optimization::pass {

/// @brief Partition every operator as a `Unigraph`.
struct SingleOperator {};

/// @brief Splits a graph into subgraphs, where each subgraph contains
///        only one operator.
/// @param arg0 An unpartitioned graph.
/// @return A vector of individual subgraphs.
Vec<std::pair<Unigraph, SingleOperator>> split_each(Unigraph &&);

} // namespace optimization::pass
