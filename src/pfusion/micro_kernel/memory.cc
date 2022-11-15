#include "pfusion/micro_kernel/memory.h"

namespace memb {

std::string MemoryOp::generateWithCond() {
    IT_ASSERT(cond.size() == 3);
    std::string code = "";
    int edge_length = cond[0] % cond[2];
    int edge_num = edge_length / width;
    int edge_width = edge_length % width;

    if (edge_num > 0 || edge_width > 0) {
        code += "if (loop_idx % " + std::to_string(cond[1]) +
                " == " + std::to_string(cond[1] - 1) + ") {\n";
    }

    if (edge_num > 0) {
        code += "#pragma unroll\n";
        code += "for (int inst_idx = 0; inst_idx < " +
                std::to_string(edge_num) + "; inst_idx++) {\n";
        if ((opType == OpType::READ && getSrc()->getType() != MemType::REG &&
             getDst()->getType() == MemType::REG) ||
            (opType == OpType::WRITE && getSrc()->getType() == MemType::REG &&
             getDst()->getType() != MemType::REG)) {
            code += getDst()->generate() + " = " + getSrc()->generate() + ";\n";
        } else {
            IT_ASSERT(false);
        }
        code += "}\n";
    }

    if (edge_width > 0) {
        code += "if (lane_id < " + std::to_string(edge_width) + ") {";
        if ((opType == OpType::READ && getSrc()->getType() != MemType::REG &&
             getDst()->getType() == MemType::REG) ||
            (opType == OpType::WRITE && getSrc()->getType() == MemType::REG &&
             getDst()->getType() != MemType::REG)) {
            code += getDst()->generateWithInstIdx(std::to_string(edge_num)) +
                    " = " +
                    getSrc()->generateWithInstIdx(std::to_string(edge_num)) +
                    ";\n";
        } else {
            IT_ASSERT(false);
        }
        code += "}\n";
    }

    if (edge_num > 0 || edge_width > 0) {
        code += "} else {\n";
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
    if (edge_num > 0 || edge_width > 0) {
        code += "}\n";
    }
    code += "// test\n";
    return code;
}

std::string MemoryOp::generate() {
    if (cond.size() != 0) {
        return generateWithCond();
    }
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