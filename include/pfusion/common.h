#pragma once

#include "core/common.h"

#include <iostream>
#include <memory>
#include <string>
#include <vector>

namespace memb {

enum OpType {
    READ = 1,
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

} // namespace memb
