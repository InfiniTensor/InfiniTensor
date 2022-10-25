#include "pfusion/micro_kernel/unary.h"

namespace memb {
std::string UnaryOp::generate() {
    std::string code;

    if (width < 32) {
        code += "if (lane_id < " + std::to_string(width) + ") {\n";
    }

    code += "#pragma unroll\n";
    code += "for (int inst_idx = 0; inst_idx < " + std::to_string(num) +
            "; inst_idx++) {\n";
    if (opType == RELU) {
        code += dst->generate() + " = (" + src->generate() + " > 0) ? " +
                src->generate() + " : 0;\n";
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
