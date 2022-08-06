#include "core/graph.h"
#include "core/kernel.h"

namespace it {

class RunEngine {
  public:
    RunEngine(Device device) : device(device) {}
    ~RunEngine() {}

    void run(Graph graph) const {
        const auto &kernelRegistry = KernelRegistry::getInstance();
        for (auto &op : graph->getOperators()) {
            // HACK: set correct data type
            Kernel *kernel = kernelRegistry.getKernel(device, op->getOpType(),
                                                      DataType::Int32);
            kernel->compute(op);
        }
    }

  private:
    Device device;
};
} // namespace it