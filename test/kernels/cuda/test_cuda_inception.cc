#include "core/graph.h"
#include "core/kernel.h"
#include "core/runtime.h"
#include "cuda/cuda_runtime.h"
#include "cuda/cuda_utility.h"
#include "operators/concat.h"
#include "operators/conv.h"
#include "operators/pooling.h"
#include "operators/unary.h"

#include "test.h"

namespace infini {

TEST(CUDA_Inception_v3_block, run) {
    const int bs = 1, initialChannels = 192, h = 32;

    auto cudaRuntime = make_ref<CudaRuntimeObj>();
    auto g = make_ref<GraphObj>(cudaRuntime);
    auto blockInput = g->addTensor({bs, initialChannels, h, h});
    vector<vector<tuple<bool, int, int>>> configs =
        // <isConv, f, r/s>
        {
            {{true, 64, 1}}, // a chain with one Conv
            {{true, 48, 1}, {true, 64, 5}},
            {{true, 64, 1}, {true, 96, 3}, {true, 96, 3}},
            {{false, 192, 3}, {true, 32, 3}},
        };
    TensorVec outputs;
    vector<OpVec> ops;
    auto maxpool =
        g->addOp<MaxPoolObj>(blockInput, nullptr, 3, 3, 1, 1, 1, 1, 1, 1);
    auto chainInput = maxpool->getOutput();
    for (auto &pathConfig : configs) {
        int inputChannels = initialChannels;
        auto input = chainInput;
        ops.emplace_back();
        for (auto &[isConv, f, r] : pathConfig) { // OpConfig
            if (isConv) {
                { // Add Conv
                    auto w = g->addTensor({f, inputChannels, r, r});
                    auto conv =
                        g->addOp<ConvObj>(input, w, nullptr, r / 2, r / 2);
                    input = conv->getOutput();
                    ops.back().emplace_back(conv);
                }
                { // Add Relu
                    auto relu = g->addOp<ReluObj>(input, nullptr);
                    input = relu->getOutput();
                    ops.back().emplace_back(relu);
                }
                inputChannels = f;
            } else { // Add AveragePool
                auto pool = g->addOp<AvgPoolObj>(input, nullptr, r, r, 1, 1,
                                                 r / 2, r / 2, 1, 1);
                input = pool->getOutput();
                ops.back().emplace_back(pool);
            }
        }
        outputs.emplace_back(input);
    }
    auto concat = g->addOp<ConcatObj>(outputs, nullptr, 1);
    g->print();

    // check connection
    EXPECT_EQ(maxpool->getSuccessors().size(), 4u);
    EXPECT_EQ(chainInput->getTargets().size(), 4u);
    for (const auto &chainOps : ops) {
        for (size_t i = 1; i < chainOps.size(); i++) {
            auto prev = chainOps[i - 1];
            auto cur = chainOps[i];
            EXPECT_EQ(prev->getSuccessors().size(), 1u);
            EXPECT_EQ(cur->getPredecessors().size(), 1u);
            EXPECT_EQ(prev->getSuccessors()[0], cur);
            EXPECT_EQ(prev, cur->getPredecessors()[0]);
        }
    }
    EXPECT_EQ(concat->getPredecessors().size(), 4u);

    // TODO: check outputs
    g->dataMalloc();
    cudaRuntime->run(g);
}
}; // namespace infini
