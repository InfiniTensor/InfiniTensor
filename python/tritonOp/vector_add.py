import triton
import triton.language as tl
import torch

@triton.jit
def kernel_vector_addition(a_ptr, b_ptr, out_ptr,
                           num_elems: tl.constexpr,
                           block_size: tl.constexpr,):

    pid = tl.program_id(axis=0)
    # tl.device_print("pid", pid)
    block_start = pid * block_size # 0 * 2 = 0, 1 * 2 = 2,
    thread_offsets = block_start + tl.arange(0, block_size)
    mask = thread_offsets < num_elems
    a_pointers = tl.load(a_ptr + thread_offsets, mask=mask)
    b_pointers = tl.load(b_ptr + thread_offsets, mask=mask)
    res = a_pointers + b_pointers
    tl.store(out_ptr + thread_offsets, res, mask=mask)


def ceil_div(x: int,y: int) -> int:
    return (x + y - 1) // y

def vector_add(a: list, b: list) -> torch.Tensor:
    # 将输入列表转换为 PyTorch 张量
    a_tensor = torch.tensor(a, device='cuda')
    b_tensor = torch.tensor(b, device='cuda')

    output_buffer = torch.empty_like(a_tensor)
    assert a_tensor.is_cuda and b_tensor.is_cuda
    num_elems = a_tensor.numel()
    assert num_elems == b_tensor.numel() # todo - handel mismatched sizes

    block_size = 1024
    grid_size = ceil_div(num_elems, block_size)
    grid = (grid_size,)
    num_warps = 8

    k2 = kernel_vector_addition[grid](a_tensor, b_tensor, output_buffer,
                                      num_elems,
                                      block_size,
                                      num_warps=num_warps
                                      )
    return output_buffer

def main():
    # 创建两个随机的 PyTorch 张量
    a = [1, 2, 3]
    b = [4, 5, 6]

    # 调用 vector_add 函数进行向量加法
    result = vector_add(a, b)

    # 打印结果
    print("Result of vector addition:", result)

    # 验证结果
    a_tensor = torch.tensor(a, device='cuda')
    b_tensor = torch.tensor(b, device='cuda')
    assert torch.allclose(result, a_tensor + b_tensor), "The result of vector addition is incorrect"

if __name__ == "__main__":
    main()