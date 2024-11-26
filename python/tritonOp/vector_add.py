import triton
import triton.language as tl
# 定义向量加法的Triton内核函数
@triton.jit
def vector_add_kernel(a_ptr, b_ptr, c_ptr, n: tl.constexpr):
    for i in tl.arange(0, n):
        c_ptr[i] = a_ptr[i] + b_ptr[i]
# 定义Python函数来启动内核
def vector_add(a, b):
    # 确保输入是PyTorch张量
    a = triton.tensor(a)
    b = triton.tensor(b)
    # 创建输出张量
    c = triton.tensor(a.shape, dtype=a.dtype)
    # 执行内核函数
    vector_add_kernel[triton.cdiv(a.numel(), 256), 256](a, b, c, a.numel())
    return c
