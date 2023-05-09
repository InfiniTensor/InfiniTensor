from InfiniTest.accuracy import Accuracy
from InfiniTest.dump import Dump
from InfiniTest.pytorch_operator import AddPytorch
from InfiniTest.performance import Profiling
from InfiniTest.pytorch_operator import AddPytorch
import torch
import numpy

a = numpy.array([1,2,3])
b = a
acc = Accuracy()
acc.computeDifference0("passed test case", a, b)

pro = Profiling()
@pro.hostProfilingWrapper(times=2)
def func():
    dump = Dump()
    dump.dumpData(case="case", input_data=a, precision=2)

@pro.hostProfilingWrapper(times=1)
def test_proto():
    input1= torch.randint(0,100,(3,4),dtype=torch.int32)
    input2= torch.randint(0,100,(3,4),dtype=torch.int32)
    output = torch.add(input1,input2)
    add = AddPytorch([input1,input2],[output])
    add.saveToFile("./prototxt")
    check = AddPytorch()
    inputs, outputs = check.loadFromFile("./prototxt")
    acc = Accuracy()
    acc.computeDifference0("a pass case", output.numpy(), outputs[0].numpy())


func()
test_proto()


