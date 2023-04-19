#include "core/blob.h"
#include "core/dummy_mutator.h"
#include "core/graph.h"
#include "core/runtime.h"
#include "core/search_engine.h"
#include "ffi/ffi_callback.h"
#include "nnet/nmutator.h"
#include "operators/conv.h"
#include "operators/unary.h"
#include "test.h"
#include <pybind11/stl.h>

#ifdef USE_CUDA
#include "cuda/cuda_runtime.h"
#endif

namespace infini {

// NHWC format
Graph getInfoGAN(int batch, Runtime runtime, int nLayers) {
    IT_ASSERT(1 <= nLayers && nLayers <= 5);
    Graph g = make_ref<GraphObj>(runtime);
    vector<Tensor> weights;
    vector<tuple<int, int, int, int, bool>> cs{
        // Channel, kernelSize, pad, stride, isTanh
        {448, 2, 0, 1, false}, {256, 4, 1, 2, false}, {128, 4, 1, 2, false},
        {64, 4, 1, 2, false},  {3, 4, 1, 2, true},
    };

    Tensor input =
        g->addTensor({batch, 1, 1, 228}, DataType::Float32, TensorType::Input);
    for (int i = 0; i < (int)cs.size() && i < nLayers; ++i) {
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

Graph optimizeGraph(Graph g, Runtime _runtime, bool tuning) {
    auto runtime = as<CudaRuntimeObj>(_runtime);
    Runtime cpu = NativeCpuRuntimeObj::getInstance();
    Graph gCpu = make_ref<GraphObj>(cpu);

    auto mutator =
        make_ref<NMutator>(NMutator::Mode::RuleBased,
                           vector<int>{3, 2, 2, 2, 2, 5, 8, 8, 6, 91, 90});
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
    dbg("Baseline graph");
    printGraph(g);
    dbg(runtime->getPerfTime(g, true));

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
            dbg(runtime->timeWithCudaGraph(bestGraph));
        }

        dbg("Best graph");
        printGraph(bestGraph);
        return bestGraph;
    }
    return nullptr;
}

vector<Tensor> runInfoGAN(int nLayers) {
#ifdef USE_CUDA
    auto cuda = make_ref<CudaRuntimeObj>();
    Runtime cpu = NativeCpuRuntimeObj::getInstance();
    Graph gCpu = make_ref<GraphObj>(cpu);

    Graph g = getInfoGAN(1, cuda, nLayers);

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
#endif
    return {};
}

// TEST(ModelE2E, InfoGAN) { runInfoGAN(); }

} // namespace infini
