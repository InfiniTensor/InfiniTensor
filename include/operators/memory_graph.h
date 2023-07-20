#pragma once
#ifndef MEMORY_GRAPH_H
#define MEMORY_GRAPH_H

#include "core/operator.h"

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
class MemoryOpAttibutes {};

/// @brief 访存算子。
struct MemoryOperator {
    /// @brief 算子在图中的序号。
    size_t opIdx;
    /// @brief 算子类型。
    MemoryOpType opType;
    /// @brief 算子属性。
    std::unique_ptr<MemoryOpAttibutes> attributes;
    /// @brief 算子输入。
    std::vector<Inlet> inputs;
    /// @brief 算子输出。
    std::vector<Outlet> outputs;
};

/// @brief 访存图。
class MemoryGraphObj : public OperatorObj {
    /// @brief 图中所有算子。
    std::vector<MemoryOperator> ops;

  public:
    /// @brief 构造一个空的访存图。
    /// @param inputs 访存图在计算图里的输入。
    /// @param outputs 访存图在计算图里的输出。
    MemoryGraphObj(TensorVec inputs, TensorVec outputs);

    /// @brief 添加新的图输入。
    /// @param shape 访存图的输入必须是连续的。
    /// @param dt 数据类型。
    /// @return 新输入的位置。
    OutletPos pushInput(std::vector<dim_t> shape, DataType dt);

    /// @brief 添加新的图输出。
    /// @param 输出数据位置。
    void setOutput(OutletPos);

    /// @brief 添加算子。
    /// @param opType 算子类型。
    /// @param attributes 算子属性。
    /// @param inputs 算子输入。
    /// @param outputs 算子输出。
    /// @return 算子的输出位置。
    std::vector<OutletPos>
    push_op(MemoryOpType opType, std::unique_ptr<MemoryOpAttibutes> attributes,
            std::vector<Inlet> inputs, std::vector<Outlet> outputs);
};
} // namespace infini

#endif // MEMORY_GRAPH_H
