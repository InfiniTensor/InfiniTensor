import onnx
from pyinfinitensor import backend as ft
from pyinfinitensor.onnx import OnnxStub


def load_onnx(runtime, filename: str) -> ft.Graph:
    stub = OnnxStub.from_onnx(onnx.load(filename), runtime, False)
    return stub.handler.getGraph()


def run_and_evaluate(runtime, g):
    ft.initializeGraphTensors(g)
    runtime.run(g, True)
    print(f'getPerfTime = {runtime.getPerfTime(g, True, False, False)}')
    print(f'Non-ctc time = {runtime.timeNonCtcOperators(g, 10, 10)}')
    print(f'Cuda graph time = {runtime.timeWithCudaGraph(g, 10)}')


def search_depth_exp():
    runtime = ft.cuda_runtime()
    graphs = [
        (ft.getGANGraph(1, runtime, 5, 0), 'InfoGAN.bs1'),
        (ft.getLongformer(runtime, 1), 'longformer.bs1'),
    ]
    print("Figure 16")
    for original_g, name in graphs:
        print(f"=== Model {name}")
        for i in range(1, 7):
            g = ft.optimizeWithDepthConstraint(original_g, runtime, i)
            ft.initializeGraphTensors(g)
            print(f'{name} Depth = {i}: {runtime.getPerfTime(g, True, True, False)} ms')
    
def perf_test():
    runtime = ft.cuda_runtime()
    g = ft.getLongformer(runtime, 1)
    run_and_evaluate(runtime, g)

if __name__ == "__main__":
    search_depth_exp()
