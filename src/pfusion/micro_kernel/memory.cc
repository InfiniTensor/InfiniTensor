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
    if ((opType == OpType::READ && getSrc()->getType() != MemType::REG &&
         getDst()->getType() == MemType::REG) ||
        (opType == OpType::WRITE && getSrc()->getType() == MemType::REG &&
         getDst()->getType() != MemType::REG)) {
        code += getDst()->generate() + " = " + getSrc()->generate() + ";\n";
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