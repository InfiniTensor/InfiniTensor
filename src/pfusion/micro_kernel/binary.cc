#include "pfusion/micro_kernel/binary.h"

namespace memb {
std::string BinaryOp::generate() {
    std::string code;

    if (width < 32) {
        code += "if (lane_id < " + std::to_string(width) + ") {\n";
    }

    code += "#pragma unroll\n";
    code += "for (int inst_idx = 0; inst_idx < " + std::to_string(num) +
            "; inst_idx++) {\n";
    std::string opFunc = pDst->generate() + " = " + pSrc0->generate();
    if (opType == ADD) {
        opFunc += " + ";
    } else if (opType == SUB) {
        opFunc += " - ";
    } else {
        IT_ASSERT(false);
    }
    opFunc += pSrc1->generate() + ";\n";
    code += opFunc;
    code += "}\n";

    if (width < 32) {
        code += "}\n";
    }

    return code;
}

} // namespace memb
