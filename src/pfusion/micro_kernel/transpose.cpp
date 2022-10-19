#include "micro_kernel/transpose.h"
#include "common.h"

namespace memb {
std::string MicroKernelTranspose::generate(Ptr src, Ptr dst, int m,
                                           std::string lda, int n,
                                           std::string ldb) {
    assert(m == 32 && n == 32);
    CodeBuffer buf;
    buf.emit("int iter = lane_id;");
    buf.emit("for (int i = 0; i < 32; i++) {");
    buf.emit(dst.base_ptr + "[" + dst.offset + " + iter * " + ldb +
             " + lane_id] = " + src.base_ptr + "[" + src.offset +
             " + lane_id *" + lda + " + iter];");
    buf.emit("iter = (iter + 1) & 31;");
    buf.emit("}");
    return buf.toString();
}
} // namespace memb
