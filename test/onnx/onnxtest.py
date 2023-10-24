import onnx
from pyinfinitensor.onnx import OnnxStub
from pyinfinitensor import backend
stub = OnnxStub(onnx.load("/root/InfiniTensor/test/onnx/matrix/bertsquad-12.onnx"), backend.bang_runtime())
with open("optimized_bertsquad-12", "wb") as f:
    f.write(stub.to_onnx("optimized_bertsquad-12").SerializeToString())
for name, tensor in stub.inputs.items():
    print(name, tensor.shape(), tensor)
import numpy
stub.init()
for name, tensor in stub.inputs.items():
    print(name, tensor.shape(), tensor)
    input = numpy.random.random(tensor.shape()).astype(numpy.float32)
    tensor.copyin_int32(input.flatten().tolist())
stub.run()
stub.init()
for name, tensor in stub.outputs.items():
    print(name, tensor.shape(), tensor)
    print(tensor.copyout_float())
