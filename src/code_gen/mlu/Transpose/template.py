import numpy as np
import tvm
from tvm import te

input_tensor = tuple()
output_tensor = tuple()

A = te.placeholder(input_tensor, name='A')
A_ch = te.compute(output_tensor, , name='A_change')

s = te.create_schedule(A_ch.op)

block_x = te.thread_axis("blockIdx.x")
block_y = te.thread_axis("blockIdx.y")
block_z = te.thread_axis("blockIdx.z")
thread_x = te.thread_axis("threadIdx.x")
thread_y = te.thread_axis("threadIdx.y")

blockdim, threaddim = 640, 1024
n, c, h, w = s[A_ch].op.axis
nc = s[A_ch].fuse(n, c)
hw = s[A_ch].fuse(h, w)
nc_o, nc_i = s[A_ch].split(nc, factor=blockdim)
hw_o, hw_i = s[A_ch].split(hw, factor=threaddim)
s[A_ch].bind(nc_i, block_x)
s[A_ch].bind(hw_i, thread_x)
print(tvm.lower(s, [A, A_ch], simple_mode=True))

func = tvm.build(s, [A, A_ch], 'cuda')
ctx = tvm.gpu(0)
a_np = np.random.uniform(size=input_tensor).astype(A.dtype)
a = tvm.nd.array(a_np, ctx)
a_ch = tvm.nd.array(np.zeros(output_tensor, dtype=A_ch.dtype), ctx)
func(a, a_ch)
evaluator = func.time_evaluator(func.entry_name, ctx, number=10)
conv_time = evaluator(a, a_ch).mean * 1e3
tot_byte = 1
for c in input_tensor: tot_byte *= c
tot_byte = tot_byte * 4 / 1024 / 1024 / 1024 # GB
print('Convolution: %f ms, Bandwidth: %f GB/s' % (conv_time, tot_byte / conv_time * 1000 * 2))

fout = open('kernel_time.dat', 'w')
fout.write('%f' % conv_time)
fout.close()

dev_module = func.imported_modules[0]
print(dev_module)
print("----GPU code----")
print(dev_module.get_source())

