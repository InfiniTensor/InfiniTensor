import onnx
from pyinfinitensor.onnx import OnnxStub
from pyinfinitensor import backend

stub = OnnxStub(onnx.load(r"/home/yswang/winter_learn/ai_compiler/InfiniTensor/pyinfinitensor/tests/resnet50-v2-7.onnx"), backend.cpu_runtime())
with open(r"/home/yswang/winter_learn/ai_compiler/InfiniTensor/pyinfinitensor/tests/optimized.onnx", "wb") as f:
    f.write(stub.to_onnx("optimized").SerializeToString())