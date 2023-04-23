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
    auto const &op = g.operators.front();
    switch (op.op_type) {
    case OpType::Conv: {
        auto const &w_shape = op.inputs.at(1)->shape;
        auto const &dilations = op.inputs.at(2)->data.cpu_data;
        auto const &pads = op.inputs.at(3)->data.cpu_data;
        auto const &strides = op.inputs.at(4)->data.cpu_data;
        if (w_shape.rbegin()[0] == 1    // fmt: new line
            && w_shape.rbegin()[1] == 1 //
            && std::all_of(strides.begin(), strides.end(),
                           [](auto x) { return x == 1; })) {
            // 1x1 conv
        } else if (std::any_of(dilations.begin(), dilations.end(),
                               [](auto x) { return x != 1; })) {
            // dilated conv
        }
    } break;

    default:
        break;
    }

    return ans;
}
