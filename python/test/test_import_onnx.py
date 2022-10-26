from pyinfinitensor import *
from infinitensor import import_onnx


class Test_ImportOnnx:
    def test_Netname(self):
        runtime = CpuRuntimeObj.getInstance()
        graphBuilder = GraphBuilderObj(runtime)
        import_onnx(graphBuilder, '/path/to/net')
