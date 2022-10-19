#include "codegen.h"
#include "memory_operator.h"
#include "micro_kernel/transpose.h"

namespace memb {

std::string header =
    std::string("// Header code\n") + std::string("#include \"cuda.h\"\n") +
    std::string("#include \"cuda_utils.h\"\n\n") +
    std::string("#define ROUND_UP(n, b) (((n) - 1) / (b) + 1)\n") +
    std::string("#define INDEX(x, y, n) ((x) * (n) + (y))\n") +
    std::string("#define MIN(x, y) ((x) < (y) ? (x) : (y))\n\n");

std::string gen_lowest_basic(int dimn, int dimx, int dimy, int nblocks,
                             int nthreads) {
    std::string code = header;
    code += "// Kernel\n";
    code += "__global__ void kernel_tmp(float *src, float *dst) {\n";
    code += "int lane_id = threadIdx.x % 32;\n";
    code += "int warp_id = threadIdx.x / 32;\n";
    code += "int parallel_idx = blockIdx.x * " + std::to_string(nthreads / 32) +
            " + warp_id;\n";
    code += "float reg[32];\n";
    code += "__shared__ float smem[32 * 32 * 2 * " +
            std::to_string(nthreads / 32) + "];\n";

    int loop = ((dimx - 1) / 32 + 1) * ((dimy - 1) / 32 + 1);
    code += "for (int loop_idx = 0; loop_idx < " + std::to_string(loop) +
            "; loop_idx++) {\n";

    MemoryOperator dram_read;
    dram_read.memoryType = MemoryOperator::DRAM;
    dram_read.opType = MemoryOperator::READ;
    dram_read.ptr = Ptr("src", "parallel_idx * " + std::to_string(dimx * dimy));
    dram_read.num =
        "MIN(32, " + std::to_string(dimx) + " - loop_idx / 32 * 32)";
    dram_read.offset =
        "INDEX(loop_idx / 32 + inst_idx, loop_idx % 32 + lane_id, " +
        std::to_string(dimy) + ")";
    dram_read.reg = "inst_idx";
    code += dram_read.generate();

    MemoryOperator sram_write;
    sram_write.memoryType = MemoryOperator::SRAM;
    sram_write.opType = MemoryOperator::WRITE;
    sram_write.ptr = Ptr("smem", "warp_id * 32 * 32 * 2");
    sram_write.num =
        "MIN(32, " + std::to_string(dimx) + " - loop_idx / 32 * 32)";
    sram_write.offset = "inst_idx * 32 + lane_id";
    sram_write.reg = "inst_idx";
    code += sram_write.generate();

    MicroKernelTranspose transpose;
    code += transpose.generate(Ptr("smem", "warp_id * 32 * 32 * 2"),
                               Ptr("smem", "warp_id * 32 * 32 * 2 + 32 * 32"),
                               32, "32", 32, "32");

    MemoryOperator sram_read;
    sram_read.memoryType = MemoryOperator::SRAM;
    sram_read.opType = MemoryOperator::READ;
    sram_read.ptr = Ptr("smem", "warp_id * 32 * 32 * 2 + 32 * 32");
    sram_read.num =
        "MIN(32, " + std::to_string(dimx) + " - loop_idx / 32 * 32)";
    sram_read.offset = "inst_idx * 32 + lane_id";
    sram_read.reg = "inst_idx";
    code += sram_read.generate();

    MemoryOperator dram_write;
    sram_read.memoryType = MemoryOperator::DRAM;
    sram_read.opType = MemoryOperator::WRITE;
    sram_read.ptr = Ptr("dst", "parallel_idx * " + std::to_string(dimx) +
                                   " * " + std::to_string(dimy));
    sram_read.num =
        "MIN(32, " + std::to_string(dimx) + " - loop_idx / 32 * 32)";
    sram_read.offset =
        "INDEX(loop_idx / 32 + inst_idx, loop_idx % 32 + lane_id, " +
        std::to_string(dimy) + ")";
    sram_read.reg = "inst_idx";
    code += sram_read.generate();
    code += "}\n";
    code += "}\n\n";

    code += "void transpose(float *src, float *dst) {\n";
    code += "dim3 gridDim(" + std::to_string(nblocks) + ", 1);";
    code += "dim3 blockDim(" + std::to_string(nthreads) + ", 1);";
    code += "kernel_tmp<<<gridDim, blockDim>>>(src, dst);\n";
    code += "cudaCheckError();\n";
    code += "}\n";

    return code;
}

} // namespace memb