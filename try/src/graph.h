﻿#pragma once

#include <unordered_set>
#include <vector>

#include "data_type.h"
#include "op_type.h"

/// @brief Stores tensor data。
struct Data {
    /// @brief `cpu_data` is stored in the memory space,
    /// which allows it to be managed using `std::vector<uint8_t>`.
    std::vector<uint8_t> cpu_data;
    // #ifdef USE_CUDA
    //     void *gpu_data;
    // #endif
    // #ifdef USE_BANG
    //     void *mlu_data;
    // #endif
};

/// @brief A tensor represented by which `node` it is passed to
/// and at which `slot` in inputs of that `node`.
struct InletPos {
    size_t node, slot;
};

/// @brief A tensor represented by which `node` it is generated from
/// and at which `slot` in outputs of that `node`.
struct OutletPos {
    size_t node, slot;
};

/// @brief Calculates the hash of `OutletPos`.
struct OutletPosHash {
    size_t operator()(OutletPos const &o) const {
        return o.node ^ (o.slot << 1);
    }
};

/// @brief The data structure represents a `Outlet` of a operator,
/// which generates a tensor, and it is part of the `Node`.
/// @tparam Tensor discripter.
template <class Tensor> struct Outlet {
    Tensor info;
    std::vector<InletPos> targets;
};

/// @brief The specific tensor information excludes all unknowns.
/// This struct can be used as a tensor discripter type in templates.
struct TensorInfo {
    std::vector<size_t> shape;
    DataType data_type;
    Data data;
};

/// @brief Operator `Node` of the dataflow `Graph`.
/// @tparam Tensor discripter.
template <class Tensor> struct Node {
    OpType op_type;
    std::vector<OutletPos> inputs;
    std::vector<Outlet<Tensor>> outputs;
};

/// @brief A reference of an operator `Node` in a dataflow `Graph`。
class OpObj {
    size_t node_idx;

  public:
    explicit OpObj(size_t node_idx) : node_idx(node_idx) {}

    OutletPos operator[](size_t slot) const { return {node_idx, slot}; }
};

/// @brief The dataflow `Graph`.
/// @tparam Tensor discripter.
///
/// **NOTICE** Methods of a template class must be defined
/// in the same file as the class.
template <class Tensor> class Graph {
    /// @brief `operators` must be topo sorted.
    std::vector<Node<Tensor>> _operators;

  public:
    /// @brief Pushs a new operator `Node` into `Graph`.
    /// @param op_type Operator type.
    /// @param inputs Tensors passed to operator.
    /// @param outputs Tensors generated by operator.
    /// @return A reference to the `Node` in `Graph`.
    OpObj push_operator(                    // fmt: new line
        OpType op_type,                     //
        std::vector<OutletPos> inputs,      //
        std::vector<Outlet<Tensor>> outputs //
    ) {
        auto index = _operators.size();

        for (const auto &input : inputs)
            if (input.node >= index)
                throw "input node not exist";

        size_t i = 0;
        for (const auto &input : inputs)
            _operators[input.node]   // fmt: new line
                .outputs[input.slot] //
                .targets             //
                .emplace_back(index, ++i);

        _operators.emplace_back(op_type, std::move(inputs), std::move(outputs));
        return OpObj(index);
    }

    /// @brief Gets operators in the `Graph`.
    /// @return Operators in the `Graph`.
    std::vector<Node<Tensor>> const &operators() const { return _operators; }

    /// @brief `Graph` inputs.
    /// @return Indices of input `Node`s in `Graph`.
    std::vector<size_t> inputs() const {
        std::vector<size_t> ans;
        size_t i = 0;
        for (const auto &node : operators) {
            if (node.op_type == OpType::Input && !node.outputs[0].data.cpu_data)
                ans.push_back(i);
            ++i;
        }
        return ans;
    }

    /// @brief `Graph` outputs.
    /// @return Indices of output `Node`s in `Graph`.
    std::vector<size_t> outputs() const {
        std::vector<size_t> ans;
        size_t i = 0;
        for (const auto &node : operators) {
            if (node.op_type == OpType::Output)
                ans.push_back(i);
            ++i;
        }
        return ans;
    }
};
