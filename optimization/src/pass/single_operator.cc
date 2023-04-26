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
    switch (g.operators.front().op_type) {
    case OpType::Conv: {
        auto const conv = Conv(g.operators.front());
        auto const &i_shape = conv.input()->shape;
        auto const &k_shape = conv.kernel()->shape;
        auto const &dilations = conv.dilations()->to_vec<int64_t>();
        auto const &strides = conv.strides()->to_vec<int64_t>();
        // assert(conv.input()->data_type == conv.kernel()->data_type);
        auto const dt = conv.input()->data_type;
        if (k_shape.rbegin()[0] == 1    // fmt: new line
            && k_shape.rbegin()[1] == 1 //
            && i_shape[1] == k_shape[1] // group = 1
            && std::all_of(strides.begin(), strides.end(),
                           [](auto x) { return x == 1; })) {
            // 1x1 conv
            auto &mutant = ans.emplace_back();

            // (input, "nchw"->"nhwc") -|transpose|-> tranposed -|reshape|-> t0
            Arc<Tensor> t0;
            {
                auto [shape_, permute_] = transpose(i_shape, "nchw", "nhwc");
                auto tranposed = Tensor::share(std::move(shape_), dt, {});
                auto permutation = Tensor::share_vec(std::move(permute_));
                mutant.push_operator(OpType::Transpose,
                                     {conv.input(), std::move(permutation)},
                                     {tranposed});
                mutant.push_operator(
                    OpType::Reshape, {std::move(tranposed)},
                    {t0 = Tensor::share(
                         {shape_[0] * shape_[1] * shape_[2], shape_[3]}, dt,
                         {})});
            }

            // (kernel,"fcrs"->"cfrs") -|transpose|-> tranposed -|reshape|-> t1
            Arc<Tensor> t1;
            {
                auto [shape_, permute_] = transpose(k_shape, "fcrs", "cfrs");
                auto tranposed = Tensor::share(std::move(shape_), dt, {});
                auto permutation = Tensor::share_vec(std::move(permute_));
                mutant.push_operator(OpType::Transpose,
                                     {conv.kernel(), std::move(permutation)},
                                     {tranposed});
                mutant.push_operator(
                    OpType::Reshape, {std::move(tranposed)},
                    {t1 = Tensor::share(
                         {shape_[0], shape_[1] /* * shape_[2] * shape_[3] */},
                         dt, {})});
            }

            // (t0,t1) -|matmul|-> x -|reshape|-> t2
            auto x = Tensor::share({t0->shape[0], t1->shape[1]}, dt, {});
            mutant.push_operator(OpType::MatMul, {std::move(t0), std::move(t1)},
                                 {x});
            auto t2 = Tensor::share(
                {i_shape[0], i_shape[2], i_shape[3], k_shape[0]}, dt, {});
            mutant.push_operator(OpType::Reshape, {std::move(x)}, {t2});

            // (t2,"nhwf"->"nfhw") -|transpose|-> output
            {
                auto [shape_, permute_] = transpose(t2->shape, "nhwf", "nfhw");
                // auto tranposed = Tensor::share(std::move(shape_), dt, {});
                auto permutation = Tensor::share_vec(std::move(permute_));
                mutant.push_operator(OpType::Transpose,
                                     {std::move(t2), std::move(permutation)},
                                     {conv.output()});
            }
        } else if (
            // group = 1
            i_shape[1] == k_shape[1]
            // stride[*] = 1
            && std::all_of(strides.begin(), strides.end(),
                           [](auto x) { return x == 1; })
            // dilation[*] > 1
            && std::any_of(dilations.begin(), dilations.end(),
                           [](auto x) { return x > 1; })) {
            // dilated conv
            auto &mutant = ans.emplace_back();

            auto t0 = Tensor::share(
                {
                    i_shape[0],
                    i_shape[1],
                    i_shape[2] / dilations[0],
                    static_cast<size_t>(dilations[0]),
                    i_shape[3] / dilations[1],
                    static_cast<size_t>(dilations[1]),
                },
                dt, {});
            mutant.push_operator(OpType::Reshape, {conv.input()}, {t0});

            auto [shape_, permute_] = transpose(t0->shape, "nc1234", "n24c13");
            auto transposed = Tensor::share(shape_, dt, {});
            auto permutation = Tensor::share_vec(std::move(permute_));
            mutant.push_operator(OpType::Transpose,
                                 {std::move(t0), std::move(permutation)},
                                 {transposed});

            auto t1 = Tensor::share(
                {
                    shape_[0] * shape_[1] * shape_[2],
                    shape_[3],
                    shape_[4],
                    shape_[5],
                },
                dt, {});
            mutant.push_operator(OpType::Reshape, {std::move(transposed)},
                                 {t1});

            Vec<size_t> shape__{
                shape_[0] * shape_[1] * shape_[2],
                k_shape[1],
                conv.output()->shape[2] / shape_[1],
                conv.output()->shape[3] / shape_[2],
            };

            auto t2 = Tensor::share(shape__, dt, {});
            mutant.push_operator(OpType::Conv,
                                 {
                                     std::move(t1),
                                     conv.kernel(),
                                     Tensor::share_vec<size_t>({1, 1}),
                                     conv.pads(),
                                     conv.strides(),
                                 },
                                 {t2});
            auto t3 = Tensor::share({shape_[0], shape_[1], shape_[2],
                                     shape__[1], shape__[2], shape__[3]},
                                    dt, {});
            mutant.push_operator(OpType::Reshape, {std::move(t2)}, {t3});

            auto [shape___, permute__] =
                transpose(t3->shape, "n12chw", "nc1h2w");
            auto transposed_ = Tensor::share(shape___, dt, {});
            auto permutation_ = Tensor::share_vec(std::move(permute__));
            mutant.push_operator(OpType::Transpose,
                                 {std::move(t3), std::move(permutation_)},
                                 {transposed_});
            mutant.push_operator(OpType::Reshape, {std::move(t3)},
                                 {conv.output()});
        }
    } break;

    default:
        break;
    }

    return ans;
}
