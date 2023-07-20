#include "core/memory_graph.h"

namespace infini {
MemoryGraphObj::MemoryGraphObj() : ops(2) {
    // 输入输出固定在 0、1 号，其他算子按拓扑序。
    ops[0] = MemoryOperator{0, MemoryOpType::Input, nullptr, {}, {}};
    ops[1] = MemoryOperator{1, MemoryOpType::Output, nullptr, {}, {}};
}

OutletPos MemoryGraphObj::pushInput(std::vector<dim_t> shape, DataType dt) {
    // 生成连续的访存模式。
    auto memory = std::vector<MemoryMode>(shape.size());
    memory[0] = {shape[0], 1};
    for (size_t i = 1; i < shape.size(); ++i) {
        memory[i] = {shape[i], shape[i - 1] * memory[i - 1].stride};
    }
    // 填写输入。
    auto &inputs = ops[0].outputs;
    auto slot = inputs.size();
    inputs.push_back({{}, std::move(memory), dt});
    return {0, slot};
}

/// @brief 检查一个访存模式是否连续。
/// @param mmm 访存模式。
/// @return 如果连续，返回 `true`。
bool check_dense(std::vector<MemoryMode> const &mmm) {
    if (mmm[0].stride != 1) {
        return false;
    }
    for (size_t i = 1; i < mmm.size(); ++i) {
        if (mmm[i].stride != mmm[i - 1].size * mmm[i - 1].stride)
            return false;
    }
    return true;
}

void MemoryGraphObj::setOutput(OutletPos pos) {
    // 输入的直接又输出不合常理。
    // TODO 但是可能需要支持空的访存图。
    assert(pos.opIdx != 0);
    // 检查访存连续性。只有连续访存的数据才能作为输出。
    auto &outlet = ops[pos.opIdx].outputs[pos.slot];
    assert(check_dense(outlet.memoryMode));
    // 填写输出。
    auto &outputs = ops[1].inputs;
    auto slot = outputs.size();
    outlet.targets.push_back({1, slot});
    outputs.push_back(Inlet{pos, outlet.memoryMode});
}

std::vector<OutletPos> MemoryGraphObj::push_op(
    MemoryOpType opType, std::unique_ptr<MemoryOpAttibutes> attributes,
    std::vector<Inlet> inputs, std::vector<Outlet> outputs) {
    auto idx = ops.size();
    auto outputSize = outputs.size();
    assert(outputSize > 0);
    ops.push_back(MemoryOperator{
        idx,
        opType,
        std::move(attributes),
        std::move(inputs),
        std::move(outputs),
    });
    std::vector<OutletPos> ans(outputSize, {idx, 0});
    for (size_t i = 0; i < outputSize; ++i) {
        ans[i].slot = i;
    }
    return ans;
}
} // namespace infini
