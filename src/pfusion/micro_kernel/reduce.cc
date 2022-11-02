#include "pfusion/micro_kernel/reduce.h"

namespace memb {
std::string ReduceOp::generate() {
    std::string code;
    IT_ASSERT(width == 32);

    code += getBuf()->generate() + " = 0;";
    code += "#pragma unroll\n";
    code += "for (int inst_idx = 0; inst_idx < " + std::to_string(num) +
            "; inst_idx++) {\n";
    std::string opFunc = getBuf()->generate() + " = " + getBuf()->generate();
    if (opType == REDUCEMEAN) {
        opFunc += " + ";
    } else {
        IT_ASSERT(false);
    }
    opFunc += getSrc()->generate() + ";\n";
    code += opFunc;
    code += "}\n";

    if (opType == REDUCEMEAN) {
        code += getBuf()->generate() + " = " + getBuf()->generate() +
                " / float(" + std::to_string(num * width) + ");\n";
    }

    return code;
}

} // namespace memb
