#include "single_operator.h"
#include <numeric>

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

static Vec<size_t> transpose(Vec<size_t> const &shape,
                             Vec<size_t> const &permute) {
    Vec<size_t> ans(shape.size());
    for (auto i = 0; i < ans.size(); ++i)
        ans[i] = shape[permute[i]];
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
        auto const &i_shape = op.inputs.at(0)->shape;
        auto const &w_shape = op.inputs.at(1)->shape;
        auto const &dilations = op.inputs.at(2)->data.cpu_data;
        auto const &pads = op.inputs.at(3)->data.cpu_data;
        auto const &strides = op.inputs.at(4)->data.cpu_data;
        if (w_shape.rbegin()[0] == 1    // fmt: new line
            && w_shape.rbegin()[1] == 1 //
            && std::all_of(strides.begin(), strides.end(),
                           [](auto x) { return x == 1; })) {
            // 1x1 conv
            auto &mutant = ans.emplace_back();

            // assert(op.inputs.at(0)->data_type == op.inputs.at(1)->data_type);
            auto dt = op.inputs.at(0)->data_type;

            // (input, "nchw"->"nhwc") -|transpose|-> t0 -|reshape|-> t1
            Arc<Tensor> t1;
            {
                Vec<size_t> nhwc{0, 2, 3, 1};
                auto t0 = Tensor::share(transpose(i_shape, nhwc), dt, {});
                mutant.push_operator(OpType::Transpose,
                                     {op.inputs.at(0), Tensor::share_vec(nhwc)},
                                     {t0});
                t1 = Tensor::share(
                    {i_shape[0] * i_shape[2] * i_shape[3], i_shape[1]}, dt, {});
                mutant.push_operator(OpType::Reshape, {std::move(t0)}, {t1});
            }

            // (weight,"fcrs"->"cfrs") -|transpose|-> t2 -|reshape|-> t3
            Arc<Tensor> t3;
            {
                Vec<size_t> cfrs{1, 0, 2, 3};
                auto t2 = Tensor::share(transpose(w_shape, cfrs), dt, {});
                mutant.push_operator(OpType::Transpose,
                                     {op.inputs.at(1), Tensor::share_vec(cfrs)},
                                     {t2});
                t3 = Tensor::share(
                    {w_shape[1], w_shape[0] * w_shape[2] * w_shape[3]}, dt, {});
                mutant.push_operator(OpType::Reshape, {std::move(t2)}, {t3});
            }

            // (t1,t3) -|matmul|-> t4
            auto t4 = Tensor::share({t1->shape[0], t3->shape[1]}, dt, {});
            mutant.push_operator(OpType::MatMul, {t1, t3}, {t4});

            // (t4,"nhwf"->"nfhw") -|transpose|-> t5 -|reshape|-> output
            {
                Vec<size_t> nfhw{0, 3, 1, 2};
                auto t5 = Tensor::share(transpose(t4->shape, nfhw), dt, {});
                mutant.push_operator(OpType::Transpose,
                                     {t4, Tensor::share_vec(nfhw)}, {t5});
                mutant.push_operator(OpType::Reshape, {std::move(t5)},
                                     {op.outputs.at(0)});
            }
        } else if (std::any_of(dilations.begin(), dilations.end(),
                               [](auto x) { return x != 1; })) {
            // dilated conv
            auto &mutant = ans.emplace_back();
        }
    } break;

    default:
        break;
    }

    return ans;
}
