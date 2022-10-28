#pragma once

#include "core/common.h"

#include <iostream>
#include <memory>
#include <string>
#include <vector>

namespace memb {

enum OpType {
    EMPTY = 1,
    READ,
    WRITE,
    RELU,
    ADD,
    SUB,
    TRANSPOSE,
};

enum MemType {
    DRAM = 1,
    SRAM,
    REG,
};

inline std::string getName(OpType opType) {
    switch (opType) {
    case (OpType::EMPTY):
        return "EMPTY";
    case (OpType::READ):
        return "READ";
    case (OpType::WRITE):
        return "WRITE";
    case (OpType::RELU):
        return "RELU";
    case (OpType::ADD):
        return "ADD";
    case (OpType::SUB):
        return "SUB";
    default:
        IT_ASSERT(false);
    }
    return "";
}

inline std::string getName(MemType memType) {
    switch (memType) {
    case (MemType::DRAM):
        return "DRAM";
    case (MemType::SRAM):
        return "SRAM";
    case (MemType::REG):
        return "REG";
    default:
        IT_ASSERT(false);
    }
    return "";
}

inline size_t hashAppend(size_t a, size_t b) {
    return (a * 10000019 + b * 10000079) % 2147483647;
}

} // namespace memb
