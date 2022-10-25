#include "pfusion/micro_kernel/memory.h"

namespace memb {
std::string MemoryOp::generate() {
    std::string code;

    if (width < 32) {
        code += "if (lane_id < " + std::to_string(width) + ") {\n";
    }

    code += "#pragma unroll\n";
    code += "for (int inst_idx = 0; inst_idx < " + std::to_string(num) +
            "; inst_idx++) {\n";
    if ((opType == OpType::READ && src->getType() != MemType::REG &&
         dst->getType() == MemType::REG) ||
        (opType == OpType::WRITE && src->getType() == MemType::REG &&
         dst->getType() != MemType::REG)) {
        code += dst->generate() + " = " + src->generate() + ";\n";
    } else {
        IT_ASSERT(false);
    }

    code += "}\n";
    if (width < 32) {
        code += "}\n";
    }
    return code;
}
} // namespace memb