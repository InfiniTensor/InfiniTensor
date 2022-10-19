#pragma once

#include <string>

#include "pfusion/code.h"
#include "pfusion/common.h"

namespace memb {
class MemoryOp : public MicroOp {
  public:
    enum MemoryType {
        DRAM = 1,
        SRAM,
    };

    enum OpType {
        READ = 1,
        WRITE,
    };
    MemoryType memoryType;
    OpType opType;
    Ptr ptr;
    std::string num;
    std::string reg;
    std::string offset;

    std::string generate();
    inline void print() override {
        std::cout << "memory ";
        switch (memoryType) {
        case DRAM:
            std::cout << "DRAM";
            break;
        case SRAM:
            std::cout << "SRAM";
            break;
        default:
            assert(false);
        }
        std::cout << " ";
        switch (opType) {
        case READ:
            std::cout << "READ";
            break;
        case WRITE:
            std::cout << "WRITE";
            break;
        default:
            assert(false);
        }
        std::cout << std::endl;
    }
};
} // namespace memb
