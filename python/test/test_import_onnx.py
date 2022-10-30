from pyinfinitensor import *
from infinitensor import import_onnx


class Test_ImportOnnx:
    def test_Netname(self):
        runtime = CpuRuntimeObj.getInstance()
        graphBuilder = GraphBuilderObj(runtime)
        import_onnx(graphBuilder, '/home/mazx/git/pf-models/bert.bs1.onnx')
