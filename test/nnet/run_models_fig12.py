import onnx
from pyinfinitensor import backend as ft
from pyinfinitensor.onnx import OnnxStub
from pyinfinitensor.tensorrt_backend import get_trt_time


def load_onnx(runtime, filename: str) -> ft.Graph:
    stub = OnnxStub.from_onnx(onnx.load(filename), runtime, False)
    return stub.handler.getGraph()


def run_and_evaluate(runtime, g):
    ft.initializeGraphTensors(g)
    runtime.run(g, True)
    return runtime.timeWithCudaGraph(g, 100)


def get_e2e_time(runtime, g, name: str):
    if name.startswith('resnet'):
        return get_trt_time(g)
    else:
        return run_and_evaluate(runtime, g)
        

def model_e2e_exp():
    runtime = ft.cuda_runtime()
    model_evaluation =[
        (lambda : ft.getGANGraph(1, runtime, 5, 0), 'InfoGAN.bs1'),
        (lambda : ft.getGANGraph(16, runtime, 5, 0), 'InfoGAN.bs16'),
        (lambda : ft.getGANGraph(1, runtime, 5, 1), 'DCGAN.bs1'),
        (lambda : ft.getGANGraph(16, runtime, 5, 1), 'DCGAN.bs16'),
        (lambda : ft.getFSRCNNGraph(1, runtime), "fsrcnn.bs1"),
        (lambda : ft.getFSRCNNGraph(16, runtime), "fsrcnn.bs16"),
        (lambda : load_onnx(runtime, '/mnt/auxHome/models/einnet/gcn.bs1.onnx'), 'gcn.bs1'),
        (lambda : load_onnx(runtime, '/mnt/auxHome/models/einnet/gcn.bs16.onnx'), 'gcn.bs16'),
        (lambda : load_onnx(runtime, '/mnt/auxHome/models/einnet/resnet18.bs1.onnx'), 'resnet18.bs1'),
        (lambda : load_onnx(runtime, '/mnt/auxHome/models/einnet/resnet18.bs16.onnx'), 'resnet18.bs16'),
        (lambda : load_onnx(runtime, '/mnt/auxHome/models/einnet/csrnet.bs1.onnx'), 'csrnet.bs1'),
        (lambda : load_onnx(runtime, '/mnt/auxHome/models/einnet/csrnet.bs16.onnx'), 'csrnet.bs16'),
        (lambda : ft.getLongformer(runtime, 1), 'longformer.bs1'),
        (lambda : ft.getLongformer(runtime, 16), 'longformer.bs16'),
    ]
    print("Figure 12")
    for graph_ctor, name in model_evaluation:
        original_g = graph_ctor()
        g = ft.optimizeModel(original_g, runtime, name)
        print(f'=== {name} {get_e2e_time(runtime, g, name)} ms')


def perf_test():
    # wrong time 26.6 ms
    # correct time 15 ms
    runtime = ft.cuda_runtime()
    g = ft.getLongformer(runtime, 1)
    run_and_evaluate(runtime, g)


if __name__ == "__main__":
    # perf_test() # For calibration
    model_e2e_exp()
