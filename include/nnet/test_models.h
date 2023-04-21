#ifdef USE_CUDA
#include "core/graph.h"
#include "core/runtime.h"
#include "core/search_engine.h"

namespace infini {

Graph getInfoGAN(int batch, Runtime runtime, int nLayers);
vector<Tensor> runInfoGAN(int nLayers);
Graph getConvtransposedNHWC(Runtime runtime, Shape shape, int layerId);
Graph optimizeGraph(Graph g, Runtime runtime, bool tuning);
void initializeGraphTensors(Graph g, double l, double r, bool useInt);

} // namespace infini

#endif
