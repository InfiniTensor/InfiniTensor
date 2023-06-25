import onnx
import torch
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import pyinfinitensor as pit
from pyinfinitensor import backend as ft
from pyinfinitensor.onnx import OnnxStub
from pyinfinitensor.tensorrt_backend import get_trt_time


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
    print(f'Op perf time = {runtime.getPerfTime(g, True, False, False)}')
    print(f'Graph perf time = {runtime.timeNonCtcOperators(g, 10, 10)}')
    t = runtime.timeWithCudaGraph(g, 100)
    print(f'Cuda graph time = {t}')
    return t


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


def construct_gemm(runtime, b, m, n, k, transA, transB):
    handler = ft.GraphHandler(runtime)
    input = handler.tensor([b, k, m] if transA else [b, m, k],
                           tensor_type=ft.TensorType.Input)
    w = handler.tensor([b, n, k] if transB else [b, k, n],
                       tensor_type=ft.TensorType.Initialized)
    handler.matmul(input, w, None, transA, transB, None, ft.Linear)
    return handler.getGraph()


def construct_conv(runtime, n, c, h, w, f, r, s, ph, pw, sh, sw, dh, dw, bias=False, relu=False):
    handler = ft.GraphHandler(runtime)
    # input = handler.tensor([1, 56, 32, 32], tensor_type=ft.TensorType.Input)
    # w = handler.tensor([12, 56, 1, 1], tensor_type=ft.TensorType.Initialized)
    # handler.conv(input, w, None, 0, 0, 1, 1, 1, 1)
    input = handler.tensor([n, c, h, w], tensor_type=ft.TensorType.Input)
    w = handler.tensor([f, c, r, s], tensor_type=ft.TensorType.Initialized)
    x = handler.conv(input, w, None, ph, pw, sh, sw, dh, dw)
    if bias:
        bias = handler.tensor([f, 1, 1], tensor_type=ft.TensorType.Initialized)
        x = handler.add(x, bias, None)
    if relu:
        x = handler.relu(x, None)
    return handler.getGraph()


def construct_conv_nhwc(runtime, n, c, h, w, f, r, s, pad, stride, dilation):
    handler = ft.GraphHandler(runtime)
    # input = handler.tensor([1, 56, 32, 32], tensor_type=ft.TensorType.Input)
    # w = handler.tensor([12, 56, 1, 1], tensor_type=ft.TensorType.Initialized)
    # handler.conv(input, w, None, 0, 0, 1, 1, 1, 1)
    input = handler.tensor([n, h, w, c], tensor_type=ft.TensorType.Input)
    w = handler.tensor([f, r, s, c], tensor_type=ft.TensorType.Initialized)
    handler.convNHWC(input, w, None, pad, pad, stride,
                     stride, dilation, dilation)
    return handler.getGraph()


def construct_convtranposed_nhwc(runtime, n, c, h, w, f, r, s, pad, stride, dilation):
    handler = ft.GraphHandler(runtime)
    input = handler.tensor([n, h, w, c], tensor_type=ft.TensorType.Input)
    w = handler.tensor([f, r, s, c], tensor_type=ft.TensorType.Initialized)
    handler.convtransposed2dNHWC(
        input, w, None, pad, pad, stride, stride, dilation, dilation)
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
            # runtime.run(g, True)
            # print(f'getPerfTime = {runtime.getPerfTime(g, True, True, False)}')
            # print(f'Non-ctc time = {runtime.timeNonCtcOperators(g, 10, 10)}')
            # save_onnx(g, f"opt_{name}_depth{i}.onnx")
            print(
                f'{name} Depth = {i}: {runtime.getPerfTime(g, True, True, False)} ms')


def get_e2e_time(runtime, g, name: str):
    if name.startswith('resnet'):
        return get_trt_time(g)
    else:
        return run_and_evaluate(runtime, g)


def model_e2e_exp(allow_tf32: bool):
    runtime = ft.cuda_runtime()
    runtime.setEnableTF32(allow_tf32)
    model_evaluation = [
        # (lambda: construct_conv(runtime, 1, 512, 7,
        #  7, 512, 3, 3, 1, 1, 1, 1, 1, 1), 'ResNet-conv3x3'),
        # (lambda: construct_conv(runtime, 1, 512, 7,
        #  7, 512, 3, 3, 1, 1, 1, 1, 1, 1, True, True), 'ResNet-conv3x3-BiasRelu'),
        # (lambda: construct_conv(runtime, 1, 1, 7,
        #  7, 1, 3, 3, 1, 1, 1, 1, 1, 1), 'ResNet-conv3x3-c1'),
        # (lambda: construct_conv(runtime, 1, 3, 7,
        #  7, 3, 3, 3, 1, 1, 1, 1, 1, 1), 'ResNet-conv3x3-c3'),
        # (lambda: construct_conv(runtime, 1, 32, 7,
        #  7, 32, 3, 3, 1, 1, 1, 1, 1, 1), 'ResNet-conv3x3-c32'),
        # (lambda: construct_conv(runtime, 1, 128, 7,
        #  7, 128, 3, 3, 1, 1, 1, 1, 1, 1), 'ResNet-conv3x3-c128'),
        # (lambda: ft.getGANGraph(1, runtime, 5, 0), 'InfoGAN.bs1'),
        # (lambda: ft.getGANGraph(16, runtime, 5, 0), 'InfoGAN.bs16'),
        # (lambda: ft.getGANGraph(1, runtime, 5, 1), 'DCGAN.bs1'),
        # (lambda: ft.getGANGraph(16, runtime, 5, 1), 'DCGAN.bs16'),
        # (lambda: ft.getFSRCNNGraph(1, runtime), "fsrcnn.bs1"),
        # (lambda: ft.getFSRCNNGraph(16, runtime), "fsrcnn.bs16"),
        # (lambda: load_onnx(runtime, '/mnt/auxHome/models/einnet/gcn.bs1.onnx'), 'gcn.bs1'),
        # (lambda: load_onnx(runtime, '/mnt/auxHome/models/einnet/gcn.bs16.onnx'), 'gcn.bs16'),
        (lambda: load_onnx(runtime, '/mnt/auxHome/models/einnet/resnet18.bs1.onnx'), 'resnet.bs1'),
        # (lambda: load_onnx(runtime, '/mnt/auxHome/models/einnet/resnet18.bs16.onnx'), 'resnet.bs16'),
        # (lambda: load_onnx(runtime, '/mnt/auxHome/models/einnet/csrnet.bs1.onnx'), 'csrnet.bs1'),
        # (lambda: load_onnx(runtime, '/mnt/auxHome/models/einnet/csrnet.bs16.onnx'), 'csrnet.bs16'),
        # (lambda : ft.getLongformer(runtime, 1), 'longformer.bs1'),
        # (lambda : ft.getLongformer(runtime, 16), 'longformer.bs16'),
        # (lambda : load_onnx(runtime, '/home/whj/workspace/InfiniTensor/cuda-build/efficientnet-b1_bs1.onnx'), 'efficientnet.b1'),
        # (lambda : load_onnx(runtime, '/home/whj/workspace/InfiniTensor/cuda-build/mobilenet_v2_bs1.onnx'), 'mobilenet_v2.bs1'),
    ]
    print("Figure 12")
    for graph_ctor, name in model_evaluation:
        t_orig, t_opt = 99999999, 99999999
        print(f"=== {name}")
        original_g = graph_ctor()
        # original_g = ft.convertNCHWtoNHWCModel(runtime, original_g)
        # save_onnx(original_g, f"orig_{name}.onnx")
        # print('Time:', get_e2e_time(runtime, original_g, name))
        t_orig = run_and_evaluate(runtime, original_g)
        g = ft.optimizeModel(original_g, runtime, name)
        # g = ft.optimizeGraph(original_g, runtime, False, ft.NMutatorMode.RuleBased,
        #                      [3, 2, 2, 2, 2, 5, 8, 8, 6, 91, 90]) # Convtranspose2gemm
        # g = ft.optimizeModelWithRules(original_g, runtime,
        #                               [3, 2, 2, 5, 8, 8, 6, 90])  # Conv2Gemm 
        save_onnx(g, f"opt_{name}.onnx")
        # run_and_evaluate(runtime, g)
        # print(get_e2e_time(runtime, g, name))
        t_opt = run_and_evaluate(runtime, g)
        print(
            f'=== {name} orig/opt=speedup {t_orig:.3f} {t_opt:.3f} {t_orig/t_opt:.2f}')
        verify_graphs(runtime, original_g, g)


def test_gemm_tf32(allow_tf32: bool):
    configs = [
        [1, 1024, 196, 85],
        [1, 128, 3136, 256],
        [1, 128, 784, 512],
        [1, 196, 231, 1024],
        [1, 196, 231, 21],
        [1, 196, 425, 1024],
        [1, 196, 896, 1024],
        [1, 196, 896, 128],
        [1, 2048, 49, 128],
        [1, 21, 50176, 21],
        [1, 231, 3136, 21],
        [1, 231, 3136, 256],
        [1, 256, 3136, 64],
        [1, 425, 196, 1024],
        [1, 425, 196, 85],
        [1, 425, 784, 512],
        [1, 49, 231, 2048],
        [1, 49, 231, 21],
        [1, 49, 896, 128],
        [1, 512, 784, 128],
        [1, 64, 3136, 256],
        [1, 784, 231, 21],
        [1, 784, 231, 512],
        [1, 896, 196, 128],
        [1, 896, 49, 2048],
    ]
    runtime = ft.cuda_runtime()
    runtime.setEnableTF32(allow_tf32)
    for config in configs:
        for transA, transB in ((False, False), (False, True), (True, False), (True, True)):
            s = 16
            align_config = [config[0], config[1]*16, config[2], config[3]]
            align_config = [config[0]]+[(v+s-1)//s*s for v in align_config[1:]]
            # align_config = config
            g = construct_gemm(runtime, *align_config, transA, transB)
            print(
                f"{allow_tf32} {transA} {transB} {align_config} {run_and_evaluate(runtime, g)}")


def perf_test():
    # wrong time 26.6 ms
    # correct time 15 ms
    runtime = ft.cuda_runtime()
    g = ft.getLongformer(runtime, 1)
    run_and_evaluate(runtime, g)


if __name__ == "__main__":
    # perf_test()
    for b in [False]:
        model_e2e_exp(b)
        # test_gemm_tf32(b)
