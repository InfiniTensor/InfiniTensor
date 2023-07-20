#include "core/data_type.h"
#include <cstddef>
#include <vector>

namespace infini {
using dim_t = int64_t;

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
    dim_t opIdx;
    /// @brief 数据是第几个输入。
    dim_t slot;
};

/// @brief 标记一个算子的输出的位置。
struct OutletPos {
    /// @brief 产生数据算子的序号。
    dim_t opIdx;
    /// @brief 数据是第几个输出。
    dim_t slot;
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

/// @brief 访存算子。
struct MemoryOperator {
    /// @brief 算子在图中的序号。
    size_t opIdx;
    /// @brief 算子类型。
    MemoryOpType opType;
    /// @brief 算子输入。
    std::vector<Inlet> inputs;
    /// @brief 算子输出。
    std::vector<Outlet> outputs;
};

/// @brief 访存图。
class MemoryGraphObj {
    /// @brief 图中所有算子。
    std::vector<MemoryOperator> ops;
};
} // namespace infini
