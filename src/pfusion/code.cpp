#include <iostream>

#include "code.h"
#include "memory_operator.h"

namespace memb {
std::string MemoryOperator::generate() {
    CodeBuffer buf;
    std::string inst;
    buf.emit("#pragma unroll");
    buf.emit("for (int inst_idx = 0; inst_idx < " + num + "; inst_idx++) {");
    if (opType == READ) {
        inst = "reg[inst_idx] = " + ptr.base_ptr + "[" + ptr.offset + " + " +
               offset + "];";
        buf.emit(inst);
    } else if (opType == WRITE) {
        inst = ptr.base_ptr + "[" + ptr.offset + " + " + offset +
               "] = " + "reg[inst_idx];";
        buf.emit(inst);
    } else {
        std::cout << "[ERROR]" << std::endl;
        exit(-1);
    }
    buf.emit("}");
    return buf.toString();
}
} // namespace memb