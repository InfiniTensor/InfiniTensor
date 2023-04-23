#include "single_operator.h"

using namespace optimization;
using namespace pass;

Vec<std::pair<Unigraph, SingleOperator>>
optimization::pass::split_each(Unigraph &&g) {
    Vec<std::pair<Unigraph, SingleOperator>> ans;
    for (auto &op : g.operators) {
        auto &[g, t] = ans.emplace_back();
        g.push_operator(op.op_type, op.inputs, op.outputs);
    }
    return ans;
}
