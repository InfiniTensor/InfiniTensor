#include "core/graph_factory.h"
#include "test.h"

namespace infini {

TEST(GraphFactory, ops) {
    Runtime runtime = CpuRuntimeObj::getInstance();
    { // conv without output
        GraphFactory gf = make_ref<GraphFactoryObj>(runtime);
        auto input =
            make_ref<TensorObj>(Shape{1, 3, 4, 4}, DataType::UInt32, runtime);
        auto weight =
            make_ref<TensorObj>(Shape{2, 3, 3, 3}, DataType::UInt32, runtime);
        auto conv = gf->conv(input, weight, 1, 1);
        EXPECT_EQ(conv->getOutput()->getDims(), (Shape{1, 2, 4, 4}));
    }
    { // conv with output
        GraphFactory gf = make_ref<GraphFactoryObj>(runtime);
        auto input =
            make_ref<TensorObj>(Shape{1, 3, 4, 4}, DataType::UInt32, runtime);
        auto weight =
            make_ref<TensorObj>(Shape{2, 3, 3, 3}, DataType::UInt32, runtime);
        auto output =
            make_ref<TensorObj>(Shape{1, 2, 4, 4}, DataType::UInt32, runtime);
        auto conv = gf->conv(input, weight, output, 1, 1);
    }
    { // matmul without output
        GraphFactory gf = make_ref<GraphFactoryObj>(runtime);
        auto A = make_ref<TensorObj>(Shape{1, 3, 5}, DataType::UInt32, runtime);
        auto B = make_ref<TensorObj>(Shape{1, 5, 2}, DataType::UInt32, runtime);
        auto matmul = gf->matmul(A, B);
        EXPECT_EQ(matmul->getOutput()->getDims(), (Shape{1, 3, 2}));
    }
    { // matmul with output
        GraphFactory gf = make_ref<GraphFactoryObj>(runtime);
        auto A = make_ref<TensorObj>(Shape{1, 3, 5}, DataType::UInt32, runtime);
        auto B = make_ref<TensorObj>(Shape{1, 5, 2}, DataType::UInt32, runtime);
        auto C = make_ref<TensorObj>(Shape{1, 3, 2}, DataType::UInt32, runtime);
        auto matmul = gf->matmul(A, B, C);
    }
    { // convtrans without output
        GraphFactory gf = make_ref<GraphFactoryObj>(runtime);
        auto input =
            make_ref<TensorObj>(Shape{1, 228, 1, 1}, DataType::UInt32, runtime);
        auto weight = make_ref<TensorObj>(Shape{228, 448, 2, 2},
                                          DataType::UInt32, runtime);
        auto convtrans = gf->convTrans(input, weight, 0, 0);
        EXPECT_EQ(convtrans->getOutput()->getDims(), (Shape{1, 448, 2, 2}));
    }
    { // convtrans with output
        GraphFactory gf = make_ref<GraphFactoryObj>(runtime);
        auto input =
            make_ref<TensorObj>(Shape{1, 228, 1, 1}, DataType::UInt32, runtime);
        auto weight = make_ref<TensorObj>(Shape{228, 448, 2, 2},
                                          DataType::UInt32, runtime);
        auto output =
            make_ref<TensorObj>(Shape{1, 448, 2, 2}, DataType::UInt32, runtime);
        auto convtrans = gf->convTrans(input, weight, 0, 0);
    }
    { // pad without output
        GraphFactory gf = make_ref<GraphFactoryObj>(runtime);
        auto input = make_ref<TensorObj>(Shape{1, 64, 162, 162},
                                         DataType::UInt32, runtime);
        vector<int> pads = {2, 10, 1, 5, 0, 10, 1, 5};
        auto pad = gf->pad(input, pads, std::nullopt);
        EXPECT_EQ(pad->getOutput()->getDims(), (Shape{3, 84, 164, 172}));
    }
    { // pad with output
        GraphFactory gf = make_ref<GraphFactoryObj>(runtime);
        auto input = make_ref<TensorObj>(Shape{1, 64, 162, 162},
                                         DataType::UInt32, runtime);
        auto output = make_ref<TensorObj>(Shape{3, 84, 164, 172},
                                          DataType::UInt32, runtime);
        vector<int> pads = {2, 10, 1, 5, 0, 10, 1, 5};
        auto pad = gf->pad(input, output, pads, std::nullopt);
    }
    { // slice without output
        GraphFactory gf = make_ref<GraphFactoryObj>(runtime);
        auto input = make_ref<TensorObj>(Shape{10, 64, 162, 162},
                                         DataType::UInt32, runtime);
        vector<int> starts = {2, 10, 1, 5};
        vector<int> ends = {3, 10, 100, 100};
        auto slice = gf->slice(input, starts, ends, std::nullopt, std::nullopt);
        EXPECT_EQ(slice->getOutput()->getDims(), (Shape{2, 1, 100, 96}));
    }
    { // slice with output
        GraphFactory gf = make_ref<GraphFactoryObj>(runtime);
        auto input = make_ref<TensorObj>(Shape{10, 64, 162, 162},
                                         DataType::UInt32, runtime);
        auto output = make_ref<TensorObj>(Shape{2, 1, 100, 96},
                                          DataType::UInt32, runtime);
        vector<int> starts = {2, 10, 1, 5};
        vector<int> ends = {3, 10, 100, 100};
        auto slice =
            gf->slice(input, output, starts, ends, std::nullopt, std::nullopt);
    }
    { // concat without output
        GraphFactory gf = make_ref<GraphFactoryObj>(runtime);
        auto t1 =
            make_ref<TensorObj>(Shape{1, 3, 2, 4}, DataType::Float32, runtime);
        auto t2 =
            make_ref<TensorObj>(Shape{1, 3, 2, 5}, DataType::Float32, runtime);
        auto concat = gf->concat(TensorVec{t1, t2}, 3);
        EXPECT_EQ(concat->getOutput()->getDims(), (Shape{1, 3, 2, 9}));
    }
    { // concat with output
        GraphFactory gf = make_ref<GraphFactoryObj>(runtime);
        auto t1 =
            make_ref<TensorObj>(Shape{1, 3, 2, 4}, DataType::Float32, runtime);
        auto t2 =
            make_ref<TensorObj>(Shape{1, 3, 2, 5}, DataType::Float32, runtime);
        auto o0 =
            make_ref<TensorObj>(Shape{1, 3, 2, 9}, DataType::Float32, runtime);
        auto concat = gf->concat(TensorVec{t1, t2}, o0, 3);
    }
    { // split without output
        GraphFactory gf = make_ref<GraphFactoryObj>(runtime);
        auto input =
            make_ref<TensorObj>(Shape{1, 3, 2, 15}, DataType::Float32, runtime);
        auto split = gf->split(input, 3, 4);
        EXPECT_EQ(split->numOutputs(), 4);
        EXPECT_EQ(split->getOutputs().size(), (size_t)4);
        EXPECT_EQ(split->getOutput(0)->getDims(), (Shape{1, 3, 2, 3}));
        EXPECT_EQ(split->getOutput(1)->getDims(), (Shape{1, 3, 2, 3}));
        EXPECT_EQ(split->getOutput(2)->getDims(), (Shape{1, 3, 2, 3}));
        EXPECT_EQ(split->getOutput(3)->getDims(), (Shape{1, 3, 2, 6}));
    }
    { // split with output
        GraphFactory gf = make_ref<GraphFactoryObj>(runtime);
        auto input =
            make_ref<TensorObj>(Shape{1, 3, 2, 15}, DataType::Float32, runtime);
        auto output0 =
            make_ref<TensorObj>(Shape{1, 3, 2, 3}, DataType::Float32, runtime);
        auto output1 =
            make_ref<TensorObj>(Shape{1, 3, 2, 3}, DataType::Float32, runtime);
        auto output2 =
            make_ref<TensorObj>(Shape{1, 3, 2, 3}, DataType::Float32, runtime);
        auto output3 =
            make_ref<TensorObj>(Shape{1, 3, 2, 6}, DataType::Float32, runtime);
        auto split = gf->split(
            input, TensorVec{output0, output1, output2, output3}, 3, 4);
    }
    { // extend without output
        GraphFactory gf = make_ref<GraphFactoryObj>(runtime);
        auto input =
            make_ref<TensorObj>(Shape{2, 3, 3, 4}, DataType::UInt32, runtime);
        auto extend = gf->extend(input, 2, 1);
        EXPECT_EQ(extend->getOutput()->getDims(), (Shape{2, 3, 6, 4}));
    }
    { // extend with output
        GraphFactory gf = make_ref<GraphFactoryObj>(runtime);
        auto input =
            make_ref<TensorObj>(Shape{2, 3, 3, 4}, DataType::UInt32, runtime);
        auto output =
            make_ref<TensorObj>(Shape{2, 3, 6, 4}, DataType::UInt32, runtime);
        auto extend = gf->extend(input, output, 2, 1);
    }
    { // maxpool without output
        GraphFactory gf = make_ref<GraphFactoryObj>(runtime);
        auto input = make_ref<TensorObj>(Shape{1, 64, 162, 162},
                                         DataType::UInt32, runtime);
        const int kh = 3, kw = 3, dh = 1, dw = 1, ph = 0, pw = 0, sh = 2,
                  sw = 2;
        auto maxpool = gf->maxpool(input, kh, kw, dh, dw, ph, pw, sh, sw);
        EXPECT_EQ(maxpool->getOutput()->getDims(), (Shape{1, 64, 80, 80}));
    }
    { // maxpool with output
        GraphFactory gf = make_ref<GraphFactoryObj>(runtime);
        auto input = make_ref<TensorObj>(Shape{1, 64, 162, 162},
                                         DataType::UInt32, runtime);
        auto output = make_ref<TensorObj>(Shape{1, 64, 80, 80},
                                          DataType::UInt32, runtime);
        const int kh = 3, kw = 3, dh = 1, dw = 1, ph = 0, pw = 0, sh = 2,
                  sw = 2;
        auto maxpool =
            gf->maxpool(input, output, kh, kw, dh, dw, ph, pw, sh, sw);
    }
    { // add without output
        GraphFactory gf = make_ref<GraphFactoryObj>(runtime);
        auto input0 =
            make_ref<TensorObj>(Shape{2, 3, 3, 4}, DataType::UInt32, runtime);
        auto input1 =
            make_ref<TensorObj>(Shape{2, 3, 3, 4}, DataType::UInt32, runtime);
        auto add = gf->add(input0, input1);
        EXPECT_EQ(add->getOutput()->getDims(), (Shape{2, 3, 3, 4}));
    }
    { // add with output
        GraphFactory gf = make_ref<GraphFactoryObj>(runtime);
        auto input0 =
            make_ref<TensorObj>(Shape{2, 3, 3, 4}, DataType::UInt32, runtime);
        auto input1 =
            make_ref<TensorObj>(Shape{2, 3, 3, 4}, DataType::UInt32, runtime);
        auto output =
            make_ref<TensorObj>(Shape{2, 3, 3, 4}, DataType::UInt32, runtime);
        auto add = gf->add(input0, input1, output);
    }
    { // gather without output
        GraphFactory gf = make_ref<GraphFactoryObj>(runtime);
        auto input =
            make_ref<TensorObj>(Shape{1, 3, 4, 4}, DataType::UInt32, runtime);
        auto index =
            make_ref<TensorObj>(Shape{2, 1, 2}, DataType::UInt32, runtime);
        auto gather = gf->gather(input, index, 1);
        EXPECT_EQ(gather->getOutput()->getDims(), (Shape{1, 2, 1, 2, 4, 4}));
    }
    { // gather with output
        GraphFactory gf = make_ref<GraphFactoryObj>(runtime);
        auto input =
            make_ref<TensorObj>(Shape{1, 3, 4, 4}, DataType::UInt32, runtime);
        auto index =
            make_ref<TensorObj>(Shape{2, 1, 2}, DataType::UInt32, runtime);
        auto output = make_ref<TensorObj>(Shape{1, 2, 1, 2, 4, 4},
                                          DataType::UInt32, runtime);
        auto gather = gf->gather(input, index, output, 1);
    }
    { // reshape without output
        GraphFactory gf = make_ref<GraphFactoryObj>(runtime);
        auto input =
            make_ref<TensorObj>(Shape{2, 3, 3, 4}, DataType::Float32, runtime);
        vector<int> dims = {3, 2, 4, 3};
        auto reshape = gf->reshape(input, dims);
        EXPECT_EQ(reshape->getOutput()->getDims(), (Shape{3, 2, 4, 3}));
    }
    { // reshape with output
        GraphFactory gf = make_ref<GraphFactoryObj>(runtime);
        auto input =
            make_ref<TensorObj>(Shape{2, 3, 3, 4}, DataType::Float32, runtime);
        vector<int> dims = {3, 2, 4, 3};
        auto output =
            make_ref<TensorObj>(Shape{3, 2, 4, 3}, DataType::Float32, runtime);
        auto reshape = gf->reshape(input, output, dims);
    }
    { // flatten without output
        GraphFactory gf = make_ref<GraphFactoryObj>(runtime);
        auto input =
            make_ref<TensorObj>(Shape{2, 3, 3, 4}, DataType::Float32, runtime);
        auto flatten = gf->flatten(input);
        EXPECT_EQ(flatten->getOutput()->getDims(), (Shape{72}));
    }
    { // flatten without output
        GraphFactory gf = make_ref<GraphFactoryObj>(runtime);
        auto input =
            make_ref<TensorObj>(Shape{2, 3, 3, 4}, DataType::Float32, runtime);
        auto output =
            make_ref<TensorObj>(Shape{72}, DataType::Float32, runtime);
        auto flatten = gf->flatten(input, output);
    }
    { // identity without output
        GraphFactory gf = make_ref<GraphFactoryObj>(runtime);
        auto input =
            make_ref<TensorObj>(Shape{2, 3, 3, 4}, DataType::Float32, runtime);
        auto identity = gf->identity(input);
        EXPECT_EQ(identity->getOutput()->getDims(), (Shape{2, 3, 3, 4}));
    }
    { // identity without output
        GraphFactory gf = make_ref<GraphFactoryObj>(runtime);
        auto input =
            make_ref<TensorObj>(Shape{2, 3, 3, 4}, DataType::Float32, runtime);
        auto output =
            make_ref<TensorObj>(Shape{2, 3, 3, 4}, DataType::Float32, runtime);
        auto identity = gf->identity(input, output);
    }
}

} // namespace infini