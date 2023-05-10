# 以下分别导入精度检测包、数据DUMP包、AddPytorch算子测试用例生成与解析类，性能度量包
from InfiniTest.accuracy import Accuracy
from InfiniTest.dump import Dump
from InfiniTest.pytorch_operator import AddPytorch
from InfiniTest.performance import Profiling
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
    input1= torch.randint(0,100,(3,4),dtype=torch.int32)
    input2= torch.randint(0,100,(3,4),dtype=torch.int32)
    # 调用算子
    output = torch.add(input1,input2)
    # 构造测例实例
    add = AddPytorch([input1,input2],[output])
    # 序列化为文件
    add.saveToFile("./prototxt")
    # 构造一个测例实例
    check = AddPytorch()
    # 反序列化测例文件，获得数据
    inputs, outputs = check.loadFromFile("./prototxt")
    # 构造一个精度检查实例
    acc = Accuracy()
    # 使用方式 0 （完全相等判断）来进行检查
    acc.computeDifference0("a pass case", output.numpy(), outputs[0].numpy())
    # 注意上述代码糅合了测例生成与测例解析两个过程，实际生产时，这两个过程应当是分开的：测例生成，测例解析&精度判定

# 另外一种使用方式，考虑到有些代码不方便以函数的形式来组装进行装饰器标记，这里提供了Start与End函数来进行其实与末尾的标记
def another_test():
    pro.hostProfilingStart()
    print("aaa")
    pro.hostProfilingEnd()

# 运行
func()
test_proto()
another_test()




