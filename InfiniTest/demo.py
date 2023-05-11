# 以下分别导入精度检测包、数据DUMP包、AddPytorch算子测试用例生成与解析类，性能度量包
from InfiniTest.accuracy import Accuracy
from InfiniTest.dump import Dump
from InfiniTest.pytorch_operator import AddPytorch
from InfiniTest.performance import Profiling
from InfiniTest import operator_pb2
# 导入 pytoch、numpy
import torch
import numpy

# 构造两个numpy 的 ndarray
a = numpy.array([1,2,3])
b = a
# 构造一个默认精度检测实例，使用方式 0 （完全相等判断）来进行检查
acc = Accuracy()
acc.computeDifference0("passed test case", a, b)
# 构造一个 Profiling 实例
pro = Profiling()

# 使用 Profiling 的 host 性能度量装饰器对func进行标注
@pro.hostProfilingWrapper(times=2)
def func():
    dump = Dump()
    dump.dumpData(case="case", input_data=a, precision=2)

# 使用 Profiling 的 host 性能度量装饰器对test_proto进行标注
@pro.hostProfilingWrapper(times=1)
def test_proto():
    # 构造tensor
    input1= torch.randint(0,100,(3,4),dtype=torch.float32)
    input2= torch.randint(0,100,(3,4),dtype=torch.float32)
    input1_gpu = input1.to(torch.device("cuda"))
    input2_gpu = input2.to(torch.device("cuda"))
    # 调用pytorch cpu, gpu算子
    output_cpu = torch.add(input1,input2)
    output_gpu = torch.add(input1_gpu, input2_gpu)
    output_gpu2cpu = output_gpu.to(torch.device("cpu"))
    # 构造测例实例
    add = AddPytorch([input1,input2],[output_cpu])
    # 序列化为文件
    add.saveToFile("./prototxt", device= operator_pb2.DEVICE_CPU, info = "pytorch 1.10, gpu h100")
    # 构造一个测例实例
    check = AddPytorch()
    # 反序列化测例文件，获得数据
    inputs, outputs = check.loadFromFile("./prototxt")
    # 构造一个精度检查实例
    acc = Accuracy()
    # 使用方式 0 （完全相等判断）来进行检查
    acc.computeDifference0("a pass case", output_cpu.numpy(), outputs[0].numpy())
    # 使用方式 1 来进行检查
    acc.computeDifference1("another pass case", output_cpu.numpy(), output_gpu2cpu.numpy())
    # 注意上述代码糅合了测例生成与测例解析两个过程，实际生产时，这两个过程应当是分开的：测例生成，测例解析&精度判定

# 另外一种使用方式，考虑到有些代码不方便以函数的形式来组装进行装饰器标记，这里提供了Start与End函数来进行起始与末尾的标记
def another_test():
    pro.hostProfilingStart()
    print("aaa")
    pro.hostProfilingEnd()

def gofusion_test():
    from pyinfinitensor.onnx import OnnxStub, backend
    from onnxsim import simplify
    import onnx
    model = onnx.load('./resnet18_untrained.onnx')
    model, check = simplify(model)
    gofusion_model = OnnxStub(model, backend.cuda_runtime())
    model = gofusion_model
    images= torch.randint(0,100,(1,3,32,32),dtype=torch.float32)
    next(model.inputs.items().__iter__())[1].copyin_float(images.reshape(-1).tolist())
    model.init()
    #模型运行两侧进行cuda性能起始标记
    pro.cudaProfilingStart()
    model.run()
    pro.cudaProfilingEnd()
    outputs = next(model.outputs.items().__iter__())[1].copyout_float()
    outputs = torch.tensor(outputs)
    outputs = torch.reshape(outputs,(1,10))

# 运行
func()
test_proto()
another_test()
gofusion_test()

