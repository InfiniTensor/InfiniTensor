#include "partitions.h"

Graph split_each(UniGraph &&g,
                 std::function<float(UniGraph const &)> const &f) {
    Graph ans;
    for (auto &op : g.operators) {
        UniGraph subgraph;
        subgraph.push_operator(op.op_type, op.inputs, op.outputs);

        Candidates candidate;
        auto score = f(subgraph);
        candidate.push(std::move(subgraph), score);

        ans.subgraphs.push_back(std::move(candidate));
    }
    return ans;
}
