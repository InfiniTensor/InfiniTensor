import backend
import onnx
import torch
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd


def to_pytorch_tensor(tensor) -> torch.Tensor:
    data = tensor.copyout_float()
    tt = torch.tensor(data)
    return tt.reshape(tensor.shape())


def run_InfoGAN(n_layers: int):
    if_tensors = backend.runInfoGAN(n_layers)
    tensors = [to_pytorch_tensor(t) for t in if_tensors]
    return tensors


def read_and_check():
    for n_layers in range(1, 6):
        ans = torch.load(f'torch_{n_layers}layers_0.pt')
        x = torch.load(f'torch_{n_layers}layers_1.pt')
        print(f'=== {n_layers} layers ===')
        print(x.abs().max())


def run():
    data = []
    for n_layers in range(5, 6):
        tensors = run_InfoGAN(n_layers)
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


if __name__ == "__main__":
    run()
    # read_and_check()
