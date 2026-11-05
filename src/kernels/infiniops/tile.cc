#include "operators/tile.h"
#include "infiniops_common.h"
#include "infiniops_dispatch.h"

#include <base/tile.h>

namespace infini {
namespace {

class TileInfiniOps : public infiniops::KernelWithoutConfig {
    void compute(const Operator &op, const RuntimeObj *context) const override {
        auto input = infiniops::toInfiniOpsTensor(op->getInputs(0), context);
        auto out = infiniops::toInfiniOpsTensor(op->getOutput(), context);
        auto handle = infiniops::makeInfiniOpsHandle(context);
        auto config = infiniops::makeInfiniOpsConfig<::infini::ops::Tile>(context);
        infiniops::dispatch::callTile(handle, config, input, out);
    }
};

} // namespace

REGISTER_KERNEL(ExecutionProvider::Infini, OpType::Tile, TileInfiniOps, "Tile_InfiniOps");

} // namespace infini
