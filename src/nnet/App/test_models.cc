#ifdef USE_CUDA
#include "core/blob.h"
#include "core/dummy_mutator.h"
#include "core/graph.h"
#include "core/runtime.h"
#include "core/search_engine.h"
#include "cuda/cuda_runtime.h"
#include "ffi/ffi_callback.h"
#include "nnet/nmutator.h"
#include "operators/conv.h"
#include "operators/unary.h"
#include "test.h"
#include <pybind11/stl.h>

namespace infini {

// Channel, kernelSize, pad, stride, isTanh
using GANConfigs = vector<tuple<int, int, int, int, bool>>;
using DetailedConfigs =
    vector<tuple<int, int, int, int, int, int, int, int, int, int, bool>>;

DetailedConfigs getGANConfigs(int id, int batch) {
    // The first conv can be transformed into gemm without reduction
    //                                       n, f,    h, w,     c, r, s, stride,
    //                                       pad, dilation
    GANConfigs ret;
    const DetailedConfigs infoConfigs = {
        {batch, 228, 1, 1, 448, 2, 2, 1, 0, 1, false},
        {batch, 448, 2, 2, 256, 4, 4, 2, 1, 1, false},
        {batch, 256, 4, 4, 128, 4, 4, 2, 1, 1, false},
        {batch, 128, 8, 8, 64, 4, 4, 2, 1, 1, false},
        {batch, 64, 16, 16, 3, 4, 4, 2, 1, 1, true}};
    const DetailedConfigs dcganConfigs = {
        {batch, 100, 1, 1, 512, 4, 4, 1, 0, 1, false},
        {batch, 512, 4, 4, 256, 4, 4, 2, 1, 1, false},
        {batch, 256, 8, 8, 128, 4, 4, 2, 1, 1, false},
        {batch, 128, 16, 16, 64, 4, 4, 2, 1, 1, false},
        {batch, 64, 32, 32, 3, 4, 4, 2, 1, 1, true}};
    DetailedConfigs details;
    if (id == 0) { // InfoGAN
        dbg("Use InfoGAN configs");
        details = infoConfigs;
    } else if (id == 1) { // DCGAN
        dbg("Use DCGAN configs");
        details = dcganConfigs;
    } else
        IT_ASSERT(false);
    return details;
}

// NHWC format
Graph getGANGraph(int batch, Runtime runtime, int nLayers, int modelId) {
    IT_ASSERT(1 <= nLayers && nLayers <= 5);
    Graph g = make_ref<GraphObj>(runtime);
    vector<Tensor> weights;
    auto configs = getGANConfigs(modelId, batch);

    Tensor input;
    {
        auto &[n, f, h, w, c, r, s, stride, pad, dilation, isTanh] = configs[0];
        input = g->addTensor({batch, 1, 1, f}, DataType::Float32,
                             TensorType::Input);
    }
    for (int i = 0; i < (int)configs.size() && i < nLayers; ++i) {
        // auto [channel, kernelSize, pad, stride, tanh] = configs[i];
        auto &[n, f, h, w, c, r, s, stride, pad, dilation, isTanh] = configs[i];
        IT_ASSERT(input->getDims()[3] == f);
        auto weight = g->addTensor({f, r, s, c}, DataType::Float32,
                                   TensorType::Initialized); // f, r, s, c
        input = g->addOp<ConvTransposed2dNHWCObj>(input, weight, nullptr, pad,
                                                  pad, stride, stride, 1, 1)
                    ->getOutput();
        if (isTanh) {
            input = g->addOp<TanhObj>(input, nullptr)->getOutput();
        } else {
            input = g->addOp<ReluObj>(input, nullptr)->getOutput();
        }
    }
    return g;
}

// NHWC
Graph getFSRCNNGraph(int batch, Runtime runtime) {
    // n, c, h, w, f, r, s, stride, pad, dilation, has_pReLU
    const DetailedConfigs fsrcnn_config = {
        {batch, 1, 32, 32, 56, 5, 5, 1, 2, 1, true},
        {batch, 56, 32, 32, 12, 1, 1, 1, 0, 1, true},
        {batch, 12, 32, 32, 12, 3, 3, 1, 1, 1, false},
        {batch, 12, 32, 32, 12, 3, 3, 1, 1, 1, false},
        {batch, 12, 32, 32, 12, 3, 3, 1, 1, 1, false},
        {batch, 12, 32, 32, 12, 3, 3, 1, 1, 1, true},
        {batch, 12, 32, 32, 56, 1, 1, 1, 0, 1, true},
        {batch, 56, 32, 32, 1, 9, 9, 1, 3, 4, false} // ConvTransNHWC
        // n, f, h, w, c, r, s, stride, pad, dilation, has_pReLU
    };

    Graph g = make_ref<GraphObj>(runtime);

    Tensor input;
    {
        auto &[n, c, h, w, f, r, s, stride, pad, dilation, has_pReLU] =
            fsrcnn_config[0];
        input = g->addTensor({batch, h, w, c}, DataType::Float32,
                             TensorType::Input);
    }

    for (int i = 0; i < (int)fsrcnn_config.size() - 1; ++i) {
        // auto [channel, kernelSize, pad, stride, tanh] = configs[i];
        auto &[n, c, h, w, f, r, s, stride, pad, dilation, has_pReLU] =
            fsrcnn_config[i];
        IT_ASSERT(input->getDims()[3] == c);
        auto weight = g->addTensor({f, r, s, c}, DataType::Float32,
                                   TensorType::Initialized); // f, r, s, c
        input = g->addOp<ConvNHWCObj>(input, weight, nullptr, pad, pad, stride,
                                      stride, 1, 1)
                    ->getOutput();
        if (has_pReLU) {
            input = g->addOp<ReluObj>(input, nullptr)->getOutput();
        }
    }

    // last operator is a ConvTransNHWC
    {
        auto &[n, f, h, w, c, r, s, stride, pad, dilation, has_pReLU] =
            fsrcnn_config[fsrcnn_config.size() - 1];
        IT_ASSERT(input->getDims()[3] == f);
        auto weight = g->addTensor({f, r, s, c}, DataType::Float32,
                                   TensorType::Initialized); // f, r, s, c
        input = g->addOp<ConvTransposed2dNHWCObj>(input, weight, nullptr, pad,
                                                  pad, stride, stride, 1, 1)
                    ->getOutput();
    }

    return g;
}

Graph getConvtransposedNHWC(Runtime runtime, Shape shape, int layerId) {
    IT_ASSERT(0 <= layerId && layerId < 5);
    Graph g = make_ref<GraphObj>(runtime);
    vector<Tensor> weights;
    vector<tuple<int, int, int, int, bool>> cs{
        // Channel, kernelSize, pad, stride, isTanh
        {448, 2, 0, 1, false}, {256, 4, 1, 2, false}, {128, 4, 1, 2, false},
        {64, 4, 1, 2, false},  {3, 4, 1, 2, true},
    };

    Tensor input = g->addTensor(shape, DataType::Float32, TensorType::Input);
    for (int i = layerId; i < layerId + 1; ++i) {
        auto [channel, kernelSize, pad, stride, tanh] = cs[i];
        int f = input->getDims()[3]; // n, h, w, f
        auto weight = g->addTensor({f, kernelSize, kernelSize, channel},
                                   DataType::Float32,
                                   TensorType::Initialized); // f, r, s, c
        input = g->addOp<ConvTransposed2dNHWCObj>(input, weight, nullptr, pad,
                                                  pad, stride, stride, 1, 1)
                    ->getOutput();
        if (tanh) {
            input = g->addOp<TanhObj>(input, nullptr)->getOutput();
        } else {
            input = g->addOp<ReluObj>(input, nullptr)->getOutput();
        }
    }
    return g;
}

void printGraph(Graph g) {
    g->print();
    puts("============ Data ============");
    for (auto t : g->getTensors()) {
        dbg(t);
        t->printData();
    }
}

void initializeGraphTensors(Graph g, double l, double r, bool useInt) {
    auto gen = RandomGenerator(-0.1, 0.1, 0, useInt);
    for (auto t : g->getInputs()) {
        t->setData(gen);
    }
    for (auto t : g->getOutputs()) {
        t->setData(ZeroGenerator());
    }
}

Graph optimizeGraph(Graph g, Runtime _runtime, bool tuning, NMutator::Mode mode,
                    vector<int> rules) {
    auto runtime = as<CudaRuntimeObj>(_runtime);
    Runtime cpu = NativeCpuRuntimeObj::getInstance();
    Graph gCpu = make_ref<GraphObj>(cpu);
    //    vector<int>{3, 2, 2, 5, 8, 8, 6, 90}); // Conv2gemm
    //    vector<int>{3, 2, 2, 2, 2, 5, 8, 8, 6, 91, 90}); // TConv
    Ref<NMutator> mutator;
    if (mode == NMutator::Mode::Normal) {
        dbg(mode);
        mutator = make_ref<NMutator>(mode);
    } else if (mode == NMutator::Mode::RuleBased) {
        dbg(mode, rules);
        IT_ASSERT_TODO(rules.size() > 0);
        mutator = make_ref<NMutator>(mode, rules);
    } else
        IT_TODO_HALT();
    vector<Graph> bestGraphs;
    SearchEngine searchEngine(runtime, mutator);
    bestGraphs.emplace_back(searchEngine.run(g));
    g->topo_sort();
    dbg(g, bestGraphs[0], bestGraphs.size());
    g->print();

    g->dataMalloc();
    map<UidBaseType, Tensor> fuidToInputTensor;
    for (auto t : g->getInputs()) {
        IT_ASSERT(fuidToInputTensor.count(t->getFuid()) == 0);
        fuidToInputTensor[t->getFuid()] = t;
    }

    auto gen = RandomGenerator(-0.1, 0.1, 0);
    for (auto t : g->getInputs()) {
        t->setData(gen);
    }
    for (auto t : g->getOutputs()) {
        t->setData(ZeroGenerator());
    }
    runtime->run(g);
    // dbg("Baseline graph");
    // printGraph(g);
    // dbg(runtme->getPerfTime(g, true));

    for (size_t i = 0; i < bestGraphs.size(); i++) {
        auto bestGraphCpu = bestGraphs[i];
        auto bestGraph =
            make_ref<GraphObj>(runtime, bestGraphCpu->getOperators());
        bestGraph->topo_sort();

        bestGraph->dataMalloc();
        // Initialize inputs with random data
        for (auto t : bestGraph->getInputs()) {
            t->copyData(fuidToInputTensor[t->getFuid()]);
        }

        // Initialize outputs with zeros
        for (auto t : bestGraph->getOutputs()) {
            t->setData(ZeroGenerator());
        }

        dbg(bestGraph);
        dbg(bestGraph->getOutputs());

        if (tuning) {
            runtime->run(bestGraph, true);  // Tune kernels
            runtime->run(bestGraph, false); // Execute transfomraed graph

            auto go0 = gCpu->cloneTensor(g->getOutputs()[0]);
            auto bgo0 = gCpu->cloneTensor(bestGraph->getOutputs()[0]);
            // EXPECT_TRUE(go0->equalData(bgo0, 1e-3));
            dbg(go0->equalData(bgo0, 1e-3));
            dbg(runtime->getPerfTime(bestGraph, true));
            dbg(runtime->timeNonCtcOperators(bestGraph));
            // dbg(runtime->timeWithCudaGraph(bestGraph));
        }

        // dbg("Best graph");
        // printGraph(bestGraph);
        return bestGraph;
    }
    return nullptr;
}

vector<Tensor> runInfoGAN(int nLayers) {
    auto cuda = make_ref<CudaRuntimeObj>();
    Runtime cpu = NativeCpuRuntimeObj::getInstance();
    Graph gCpu = make_ref<GraphObj>(cpu);

    Graph g = getGANGraph(1, cuda, nLayers, 0);

    auto mutator =
        make_ref<NMutator>(NMutator::Mode::RuleBased,
                           vector<int>{3, 2, 2, 2, 2, 5, 8, 8, 6, 91, 90});
    // // Translate OP to membound without derivation
    // mutator->setToNaiveMembound();

    vector<Graph> bestGraphs;
    SearchEngine searchEngine(cuda, mutator);
    bestGraphs.emplace_back(searchEngine.run(g));
    g->topo_sort();
    dbg(g, bestGraphs[0], bestGraphs.size());
    g->print();

    g->dataMalloc();
    map<UidBaseType, Tensor> fuidToInputTensor;
    for (auto t : g->getInputs()) {
        IT_ASSERT(fuidToInputTensor.count(t->getFuid()) == 0);
        fuidToInputTensor[t->getFuid()] = t;
    }

    auto gen = RandomGenerator(-0.1, 0.1, 0);
    // auto gen = RandomGenerator(-5, 5, 0, true);
    for (auto t : g->getInputs()) {
        t->setData(gen);
    }
    for (auto t : g->getOutputs()) {
        t->setData(ZeroGenerator());
    }
    cuda->run(g);
    dbg("Baseline graph");
    printGraph(g);
    dbg(cuda->getPerfTime(g, true));

    for (size_t i = 0; i < bestGraphs.size(); i++) {
        auto bestGraphCpu = bestGraphs[i];
        auto bestGraph = make_ref<GraphObj>(cuda, bestGraphCpu->getOperators());
        bestGraph->topo_sort();

        bestGraph->dataMalloc();
        // Initialize inputs with random data
        for (auto t : bestGraph->getInputs()) {
            t->copyData(fuidToInputTensor[t->getFuid()]);
        }

        // Initialize outputs with zeros
        for (auto t : bestGraph->getOutputs()) {
            t->setData(ZeroGenerator());
        }

        dbg(bestGraph);
        dbg(bestGraph->getOutputs());

        cuda->run(bestGraph, true);  // Tune kernels
        cuda->run(bestGraph, false); // Execute transfomraed graph

        auto go0 = gCpu->cloneTensor(g->getOutputs()[0]);
        auto bgo0 = gCpu->cloneTensor(bestGraph->getOutputs()[0]);
        // EXPECT_TRUE(go0->equalData(bgo0, 1e-3));
        std::cout << go0->equalData(bgo0, 1e-3) << std::endl;
        bgo0->printData();
        go0->printData();
        dbg(cuda->getPerfTime(bestGraph, true));

        dbg("Best graph");
        printGraph(bestGraph);
        callback::exportONNX(bestGraph, "best_graph.onnx"); // Debug
        return {g->getOutputs()[0], bestGraph->getOutputs()[0]};
    }
    return {};
}

} // namespace infini
#endif
