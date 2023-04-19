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


def runSingleConvT():
    runtime = ft.cuda_runtime()
    g = ft.getConvtransposedNHWC(runtime, [1, 2, 2, 448], 1)
    opt_g = ft.optimizeGraph(g, runtime)
    ft.if_onnx.export_onnx(opt_g, 'convtransposed.onnx')


def run_InfoGAN_without_tuning(tuning: bool):
    runtime = ft.cuda_runtime()
    g = ft.getInfoGAN(1, runtime, 5)
    # g = ft.getInfoGAN(1, runtime, 1)
    opt_g = ft.optimizeGraph(g, runtime, tuning)
    stub = OnnxStub.from_graph(opt_g)
    with open("optimized.onnx", "wb") as f:
        f.write(stub.to_onnx("optimized").SerializeToString())



if __name__ == "__main__":
    # run_e2e_InfoGAN()
    run_InfoGAN_without_tuning(True)
    # runSingleConvT()
    # read_and_check()
