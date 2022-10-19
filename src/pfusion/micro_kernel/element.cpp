#include "micro_kernel/element.h"
#include "common.h"

namespace memb {
std::string MicroKernelElement::gen_func() {
    CodeBuffer buf;
    buf.emit("__device__ float " + function_name + "(float element) {");
    buf.emit(function_code);
    buf.emit("}");
    return buf.toString();
}

std::string MicroKernelElement::gen_kernel(Ptr src, Ptr dst, int m, int n,
                                           std::string lda) {
    assert(m == 32 && n == 32);
    assert(function_name != "");
    CodeBuffer buf;
    buf.emit("for (int i = 0; i < 32; i++) {");
    buf.emit(dst.base_ptr + "[" + dst.offset + " + i * " + lda +
             " + lane_id] = " + function_name + "(" + src.base_ptr + "[" +
             src.offset + " + i *" + lda + " + lane_id]);");
    buf.emit("iter = (iter + 1) & 31;");
    buf.emit("}");
    return buf.toString();
}
} // namespace memb
