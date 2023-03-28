#include "core/graph.h"
#include "optimizations/transformations/transformation.h"

namespace infini {
class Partition {
  public:
    Partition() {}

  protected:
    enum RankingMetrics {
        RankByExecTime,
        RankByMemoryUsage,
    };

    vector<Graph> runTransformation(const Graph graph,
                                    Ref<Transformation> transformation) {
        return transformation->run(graph);
    }

    Graph getBestTransformation(const Graph graph,
                                Ref<Transformation> transformation,
                                RankingMetrics metrics = RankByExecTime) {
        return getTopKTransformations(graph, transformation, 1, metrics);
    }

    Graph getTopKTransformations(const Graph graph,
                                 Ref<Transformation> transformation, size_t k,
                                 RankingMetrics metrics = RankByExecTime);

    virtual Graph run(const Graph graph,
                      Ref<Transformation> transformation) = 0;
};
} // namespace infini