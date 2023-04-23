import onnx
import torch
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import pyinfinitensor as pit
from pyinfinitensor import backend as ft
from pyinfinitensor.onnx import OnnxStub


def to_pytorch_tensor(tensor) -> torch.Tensor:
    data = tensor.copyout_float()
    tt = torch.tensor(data)
    return tt.reshape(tensor.shape())


def save_onnx(opt_g: ft.Graph, filename: str):
    stub = OnnxStub.from_graph(opt_g)
    with open(filename, "wb") as f:
        f.write(stub.to_onnx("optimized").SerializeToString())


def load_onnx(runtime, filename: str) -> ft.Graph:
    stub = OnnxStub.from_onnx(onnx.load(filename), runtime, False)
    return stub.handler.getGraph()


def run_and_evaluate(runtime, g):
    ft.initializeGraphTensors(g)
    runtime.run(g, True)
    print(f'getPerfTime = {runtime.getPerfTime(g, True, False, False)}')
    print(f'Non-ctc time = {runtime.timeNonCtcOperators(g, 1000, 1000)}')
    print(f'Cuda graph time = {runtime.timeWithCudaGraph(g, 100)}')


def run_graph_get_output_as_torch_tensor(runtime, g):
    ft.initializeGraphTensors(g)
    runtime.run(g, True)
    runtime.run(g, False)
    tensors = [to_pytorch_tensor(t) for t in g.outputs()]
    assert len(tensors) == 1
    return tensors[0]


def compare_tensors(ans, x):
    assert ans.shape == x.shape
    print(f'Allclose {torch.allclose(ans, x)}')
    # Print error numbers
    tot = np.product(ans.shape)
    data = []
    for i in range(0, 10):
        tol = 10**(-i)
        clo = torch.isclose(ans, x, atol=tol, rtol=tol).sum().item()
        print(f'0.1^{i} close: {clo}/{tot} = {clo/tot}')
        data.append(clo/tot)

    # for i, t in enumerate(tensors):
    #     torch.save(t, f'torch_{n_layers}layers_{i}.pt')

    # rel_err = torch.abs((ans-x)/ans)
    # print(f'rel_err = {rel_err}')
    # print(f'max rel err = {rel_err.max()}')
    print(f'ans = {ans}')
    print(f'x = {x}')


def verify_graphs(runtime, g_original, g_new):
    ans = run_graph_get_output_as_torch_tensor(runtime, g_original)
    x = run_graph_get_output_as_torch_tensor(runtime, g_new)
    compare_tensors(ans, x)


def evluate_GANs():
    runtime = ft.cuda_runtime()
    for model_id in [0, 1]:
        for batch in [1, 16]:
            if True:
                original_g = ft.getGANGraph(batch, runtime, 5, model_id)
                g = ft.optimizeGraph(original_g, runtime, False, {
                                     3, 2, 2, 2, 2, 5, 8, 8, 6, 91, 90})
            else:
                g = load_onnx(runtime)
            save_onnx(
                g, f"{['infogan', 'dcgan'][model_id]}_optimized_{batch}.onnx")

            run_and_evaluate(runtime, g)


# def construct_convTranspose2d(runtime):
#     handler = ft.GraphHandler(runtime)
#     input = handler.tensor([1, 56, 32, 32], tensor_type=ft.TensorType.Input)
#     w = handler.tensor([56, 1, 9, 9], tensor_type=ft.TensorType.Initialized)
#     handler.convTransposed2d(input, w, None, 3, 3, 4, 4, 1, 1, 1, 1)
#     return handler.getGraph()

def construct_convTranspose2d(runtime, n, c, h, w, f, r, s, pad, stride, dilation):
    handler = ft.GraphHandler(runtime)
    input = handler.tensor([n, f, h, w], tensor_type=ft.TensorType.Input)
    w = handler.tensor([f, c, r, s], tensor_type=ft.TensorType.Initialized)
    handler.convTransposed2d(input, w, None, 3, 3, 4, 4, 1, 1, 1, 1)
    return handler.getGraph()


def construct_conv(runtime, n, c, h, w, f, r, s, pad, stride, dilation):
    handler = ft.GraphHandler(runtime)
    # input = handler.tensor([1, 56, 32, 32], tensor_type=ft.TensorType.Input)
    # w = handler.tensor([12, 56, 1, 1], tensor_type=ft.TensorType.Initialized)
    # handler.conv(input, w, None, 0, 0, 1, 1, 1, 1)
    input = handler.tensor([n, c, h, w], tensor_type=ft.TensorType.Input)
    w = handler.tensor([f, c, r, s], tensor_type=ft.TensorType.Initialized)
    handler.conv(input, w, None, pad, pad, stride, stride, dilation, dilation)
    return handler.getGraph()

def construct_conv_nhwc(runtime, n, c, h, w, f, r, s, pad, stride, dilation):
    handler = ft.GraphHandler(runtime)
    # input = handler.tensor([1, 56, 32, 32], tensor_type=ft.TensorType.Input)
    # w = handler.tensor([12, 56, 1, 1], tensor_type=ft.TensorType.Initialized)
    # handler.conv(input, w, None, 0, 0, 1, 1, 1, 1)
    input = handler.tensor([n, h, w, c], tensor_type=ft.TensorType.Input)
    w = handler.tensor([f, r, s, c], tensor_type=ft.TensorType.Initialized)
    handler.convNHWC(input, w, None, pad, pad, stride, stride, dilation, dilation)
    return handler.getGraph()

def construct_convtranposed_nhwc(runtime, n, c, h, w, f, r, s, pad, stride, dilation):
    handler = ft.GraphHandler(runtime)
    input = handler.tensor([n, h, w, c], tensor_type=ft.TensorType.Input)
    w = handler.tensor([f, r, s, c], tensor_type=ft.TensorType.Initialized)
    handler.convtransposed2dNHWC(input, w, None, pad, pad, stride, stride, dilation, dilation)
    return handler.getGraph()


def export_op_level_onnx(runtime):
    graphs = [
        (construct_conv(runtime, 1, 512, 7, 7, 512, 3, 3,
         1, 1, 1), "orig_Conv3x3"),  # ResNet18 Conv_37
        # 16, 256, 2, 2, 448, 4, 4, 1, 2, 1 # CelebA_ConvTranspose_0
        # TODO
        (construct_convTranspose2d(), "orig_ConvTranspose"),
        (construct_conv(runtime, 16, 32, 224, 224, 1, 5,
         5, 2, 1, 1, 1), "orig_Conv5x5"),  # SRCNN_Conv_4
        (construct_convTranspose2d(), "orig_G2BMM"),
    ]
    for g, name in graphs:
        save_onnx(g, f"opt_{name}.onnx")


if __name__ == "__main__":
    runtime = ft.cuda_runtime()
    graphs = [
        # (construct_conv(runtime, 16, 56, 32, 32, 12, 1, 1, 0, 1, 1), 'conv1x1'), # FSRCNN Conv_2 1x1
        # (construct_conv(runtime, 1, 12, 32, 32, 12, 3, 3, 1, 1, 1), 'conv3x3'),  # FSRCNN Conv_4 3x3
        # ft.getGANGraph(batch, runtime, 5, 1)
        # (ft.getLongformer(runtime, 1), 'longformer.bs1'),
        # (ft.getLongformer(runtime, 16), 'longformer.bs16'),
        # construct_convTranspose2d(runtime)
        # (load_onnx(runtime, '/mnt/auxHome/models/einnet/fsrcnn.bs1.onnx'), 'fsrcnn.bs1'),
        (ft.getFSRCNNGraph(1, runtime), "fsrcnn.bs1"),
        (ft.getFSRCNNGraph(16, runtime), "fsrcnn.bs16")
        # (construct_conv_nhwc(runtime, 1, 56, 32, 32, 12, 1, 1, 0, 1, 1), 'conv1x1')
    ]

    for original_g, name in graphs:
        print(f"=== {name}")
        if True:  # Optimization
            save_onnx(original_g, f"orig_{name}.onnx")
            g = ft.optimizeGraph(original_g, runtime, False, ft.NMutatorMode.RuleBased,
                                 [1, 7, 7, 2, 8, 6, 6])  # G2BMM/GBMM
            # g = ft.optimizeGraph(original_g, runtime, False, ft.NMutatorMode.RuleBased,
            #                      [3, 2, 2, 5, 8, 8, 6, 90]) # Conv2conv
            # g = ft.optimizeGraph(original_g, runtime, False, ft.NMutatorMode.Normal)

        save_onnx(g, f"opt_{name}.onnx")
        # verify_graphs(runtime, original_g, g)
        # run_and_evaluate(runtime, original_g)
        run_and_evaluate(runtime, g)
