#include "single_operator.h"
#include "../operator/conv.h"
#include <iterator>
#include <map>
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

// 1st: new shape
// 2nd: permutation
static std::pair<Vec<size_t>, Vec<size_t>> // fmt: new line
transpose(                                 //
    Vec<size_t> const &shape,              //
    char const *src,                       // source tensor layout
    char const *tgt                        // target tensor layout
) {
    // assert( shape.size() == str_len(src) == str_len(tgt) )
    std::map<char, size_t> indices;

    for (size_t i = 0; i < shape.size(); ++i)
        indices[src[i]] = i;

    auto ans = std::make_pair(     // fmt: new line
        Vec<size_t>(shape.size()), // shape
        Vec<size_t>(shape.size())  // permutation
    );

    for (auto i = 0; i < shape.size(); ++i)
        ans.first[i] = shape[ans.second[i] = indices[tgt[i]]];

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
        auto const conv = Conv(g.operators.front());
        auto const &i_shape = conv.input()->shape;
        auto const &w_shape = conv.weight()->shape;
        auto const &dilations = conv.delations()->data.cpu_data;
        auto const &strides = conv.strides()->data.cpu_data;
        if (w_shape.rbegin()[0] == 1    // fmt: new line
            && w_shape.rbegin()[1] == 1 //
            && std::all_of(strides.begin(), strides.end(),
                           [](auto x) { return x == 1; })) {
            // 1x1 conv
            auto &mutant = ans.emplace_back();

            // assert(conv.input()->data_type == conv.weight()->data_type);
            auto const dt = conv.input()->data_type;

            // (input, "nchw"->"nhwc") -|transpose|-> reshape -|reshape|-> t0
            Arc<Tensor> t0;
            {
                auto [shape_, permute_] = transpose(i_shape, "nchw", "nhwc");
                auto tranpose = Tensor::share(std::move(shape_), dt, {});
                auto permutation = Tensor::share_vec(std::move(permute_));
                mutant.push_operator(OpType::Transpose,
                                     {conv.input(), std::move(permutation)},
                                     {tranpose});
                mutant.push_operator(
                    OpType::Reshape, {std::move(tranpose)},
                    {t0 = Tensor::share(
                         {shape_[0] * shape_[1] * shape_[2], shape_[3]}, dt,
                         {})});
            }

            // (weight,"fcrs"->"cfrs") -|transpose|-> reshape -|reshape|-> t1
            Arc<Tensor> t1;
            {
                auto [shape_, permute_] = transpose(w_shape, "fcrs", "cfrs");
                auto tranpose = Tensor::share(std::move(shape_), dt, {});
                auto permutation = Tensor::share_vec(std::move(permute_));
                mutant.push_operator(OpType::Transpose,
                                     {conv.weight(), std::move(permutation)},
                                     {tranpose});
                mutant.push_operator(
                    OpType::Reshape, {std::move(tranpose)},
                    {t1 = Tensor::share(
                         {shape_[0], shape_[1] * shape_[2] * shape_[3]}, dt,
                         {})});
            }

            // (t0,t1) -|matmul|-> t2
            auto t2 = Tensor::share({t0->shape[0], t1->shape[1]}, dt, {});
            mutant.push_operator(OpType::MatMul, {t0, t1}, {t2});

            // (t2,"nhwf"->"nfhw") -|transpose|-> reshape -|reshape|-> output
            {
                auto [shape_, permute_] = transpose(t2->shape, "nhwf", "nfhw");
                auto tranpose = Tensor::share(std::move(shape_), dt, {});
                auto permutation = Tensor::share_vec(std::move(permute_));
                mutant.push_operator(OpType::Transpose,
                                     {std::move(t2), std::move(permutation)},
                                     {tranpose});
                mutant.push_operator(OpType::Reshape, {std::move(tranpose)},
                                     {conv.output()});
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
