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


def run_InfoGAN_return_tesnor(n_layers: int):
    if_tensors = ft.runInfoGAN(n_layers)
    tensors = [to_pytorch_tensor(t) for t in if_tensors]
    return tensors


def read_and_check():
    for n_layers in range(1, 6):
        ans = torch.load(f'torch_{n_layers}layers_0.pt')
        x = torch.load(f'torch_{n_layers}layers_1.pt')
        print(f'=== {n_layers} layers ===')
        print(x.abs().max())


def run_e2e_InfoGAN():
    data = []
    for n_layers in range(5, 6):
        tensors = run_InfoGAN_return_tesnor(n_layers)
        for i, t in enumerate(tensors):
            torch.save(t, f'torch_{n_layers}layers_{i}.pt')
        print(f'============ {n_layers} layers = = =')
        ans, x = tensors
        print(f'Allclose {torch.allclose(ans, x)}')

        # Print error numbers
        tot = np.product(ans.shape)
        data.append([])
        for i in range(0, 10):
            tol = 10**(-i)
            clo = torch.isclose(ans, x, atol=tol, rtol=tol).sum().item()
            print(f'0.1^{i} close: {clo}/{tot} = {clo/tot}')
            data[-1].append(clo/tot)

        rel_err = torch.abs((ans-x)/ans)
        print(rel_err, rel_err.max())
        print(f'ans = {ans}')
        print(f'x = {x}')

        # # Plot CDF
        # fig, axes = plt.subplots(9,1)
        # print(axes)
        # for i, ax in enumerate(axes):
        #     print(i)
        #     ax:plt.Axes
        #     ax.hist(torch.flatten(rel_err), density=True, cumulative=True, label='CDF',
        #         histtype='step', alpha=0.8, color='k')
        #     ax.set_xlim(0, 10**(-i))
        #     # ax.set_title('')
        # plt.show()
        # plt.savefig('a.pdf')
    df = pd.DataFrame(data)
    print(df.to_string())
    df.set_axis([f'0.1^{i}' for i in range(0, 10)], axis=1, inplace=True)
    print(df.to_string())
    df.to_csv('a.csv')


def getSingleConvT(runtime):
    return ft.getConvtransposedNHWC(runtime, [1, 2, 2, 448], 1)


def save_onnx(opt_g: ft.Graph, filename: str):
    stub = OnnxStub.from_graph(opt_g)
    with open(filename, "wb") as f:
        f.write(stub.to_onnx("optimized").SerializeToString())


def load_onnx(runtime, filename: str) -> ft.Graph:
    stub = OnnxStub.from_onnx(onnx.load(filename), runtime, False)
    return stub.handler.getGraph()


def run_and_evaluate(runtime, g):
    runtime.run(g, True)
    print(f'getPerfTime = {runtime.getPerfTime(g, True, False, False)}')
    print(f'Non-ctc time = {runtime.timeNonCtcOperators(g, 1000, 1000)}')
    print(f'Cuda graph time = {runtime.timeWithCudaGraph(g)}')


def run_graph_get_output(runtime, g):
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

    # rel_err = torch.abs((ans-x)/ans)
    # print(f'rel_err = {rel_err}')
    # print(f'max rel err = {rel_err.max()}')
    print(f'ans = {ans}')
    print(f'x = {x}')


def verify_graphs(runtime, g_original, g_new):
    ans = run_graph_get_output(runtime, g_original)
    x = run_graph_get_output(runtime, g_new)
    compare_tensors(runtime, ans, x)


def evluate_GANs():
    runtime = ft.cuda_runtime()
    for model_id in [0, 1]:
        for batch in [1, 16]:
            if True:
                original_g = ft.getGANGraph(batch, runtime, 5, model_id)
                g = ft.optimizeGraph(original_g, runtime, tuning=False)
            else:
                g = load_onnx(runtime)
            save_onnx(
                g, f"{['infogan', 'dcgan'][model_id]}_optimized_{batch}.onnx")

            run_and_evaluate(runtime, g)


if __name__ == "__main__":
    runtime = ft.cuda_runtime()
    # run_e2e_InfoGAN()
    # runSingleConvT()
    # read_and_check()
    for batch in [1, 16]:
        if True:
            original_g = ft.getGANGraph(batch, runtime, 5, 1)
            # original_g = ft.getConvtransposedNHWC(runtime, [1, 1, 1, 228], 0) # ConvTranspose 2x2
            # original_g = ft.getConvtransposedNHWC(runtime, [16, 2, 2, 448], 1) # ConvTranspose 4x4
            g = ft.optimizeGraph(original_g, runtime, tuning=False)
        else:
            g = load_onnx(runtime)
        save_onnx(g, f"dcgan_optimized_{batch}.onnx")

        verify_graphs(runtime, original_g, g)

        run_and_evaluate(runtime, g)
