#include "core/data_type.h"
#include <cstddef>
#include <memory>
#include <vector>

namespace infini {
using dim_t = int64_t;
using idx_t = size_t;

/// @brief 访存图的算子类型。
enum class MemoryOpType {
    Input,
    Output,
    Memory,
    Relu,
    Sigmoid,
    Tanh,
    Abs,
};

/// @brief 标记一个数据的输入的位置。
struct InletPos {
    /// @brief 接收数据算子的序号。
    idx_t opIdx;
    /// @brief 数据是第几个输入。
    idx_t slot;
};

/// @brief 标记一个算子的输出的位置。
struct OutletPos {
    /// @brief 产生数据算子的序号。
    idx_t opIdx;
    /// @brief 数据是第几个输出。
    idx_t slot;
};

/// @brief 一个维度的访存模式。
struct MemoryMode {
    /// @brief 维度长度。
    dim_t size;
    /// @brief 维度中两组数据在整个连续内存中跨度。
    dim_t stride;
};

/// @brief 算子输入。
struct Inlet {
    /// @brief 数据的来源。
    OutletPos source;
    /// @brief 数据的读法。
    std::vector<MemoryMode> memoryMode;
};

/// @brief 算子输出。
struct Outlet {
    /// @brief 数据的去处。
    std::vector<InletPos> targets;
    /// @brief 数据的写法。
    std::vector<MemoryMode> memoryMode;
    /// @brief 数据类型。
    DataType dt;
};

/// @brief 算子属性的基类。
class OpAttibutes {};

/// @brief 访存算子。
struct MemoryOperator {
    /// @brief 算子在图中的序号。
    size_t opIdx;
    /// @brief 算子类型。
    MemoryOpType opType;
    /// @brief 算子属性。
    std::unique_ptr<OpAttibutes> attributes;
    /// @brief 算子输入。
    std::vector<Inlet> inputs;
    /// @brief 算子输出。
    std::vector<Outlet> outputs;
};

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

/// @brief 访存图。
class MemoryGraphObj {
    /// @brief 图中所有算子。
    std::vector<MemoryOperator> ops;

  public:
    /// @brief 构造一个空的访存图。
    MemoryGraphObj()
        : ops{
              // 输入输出自动在 0、1 号，其他算子按拓扑序。
              MemoryOperator{0, MemoryOpType::Input, {}, {}},
              MemoryOperator{1, MemoryOpType::Output, {}, {}},
          } {}

    /// @brief 添加新的图输入。
    /// @param shape 访存图的输入必须是连续的。
    /// @param dt 数据类型。
    /// @return 新输入的位置。
    OutletPos pushInput(std::vector<dim_t> shape, DataType dt) {
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

    /// @brief 添加新的图输出。
    /// @param pos 输出数据位置。
    void setOutput(OutletPos pos) {
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
};
} // namespace infini
