#include "single_operator.h"

using namespace optimization;
using namespace pass;

Vec<std::pair<Unigraph, SingleOperator>>
optimization::pass::partition(Unigraph &&g) {
    Vec<std::pair<Unigraph, SingleOperator>> ans;
    for (auto &op : g.operators) {
        auto &[g, t] = ans.emplace_back();
        g.push_operator(op.op_type, op.inputs, op.outputs);
    }
    return ans;
}

Vec<Unigraph> optimization::pass::mutate( // fmt: new line
    Unigraph const &g,                    //
    SingleOperator const &                //
) {
    Vec<Unigraph> ans;

    switch (g.operators.front().op_type) {
    case OpType::Conv:
        /* code */
        break;

    default:
        break;
    }

    return ans;
}
