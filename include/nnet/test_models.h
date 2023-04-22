#ifdef USE_CUDA
#include "core/graph.h"
#include "core/runtime.h"
#include "core/search_engine.h"

namespace infini {

Graph getGANGraph(int batch, Runtime runtime, int nLayers, int modelId);
Graph getLongformer(Runtime runtime, int bs);
vector<Tensor> runInfoGAN(int nLayers);
Graph getConvtransposedNHWC(Runtime runtime, Shape shape, int layerId);
Graph optimizeGraph(Graph g, Runtime _runtime, bool tuning, NMutator::Mode mode,
                    vector<int> rules);
void initializeGraphTensors(Graph g, double l, double r, bool useInt);

} // namespace infini

#endif
