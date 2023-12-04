import onnx
import numpy as np
import torch
from pyinfinitensor.onnx import OnnxStub, backend
from onnx.shape_inference import infer_shapes
from onnxsim import simplify
import multiprocessing as mp


def start_workers(rank, world_size):
    if rank == 0:
        runtime = backend.CudaRuntime(rank)
        runtime.init_comm(
            "sendrecv",
            world_size,
            rank,
        )
        leftpath = "leftsendrecv.onnx"
        leftmodel = onnx.load(leftpath)
        stub = OnnxStub(leftmodel, runtime)
        input_data = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=np.float32).reshape(
            2, 3
        )
        next(iter(stub.inputs.values())).copyin_numpy(input_data)
        stub.run()
        print(rank, next(stub.inputs.values().__iter__()).copyout_numpy())
    elif rank == 2:
        runtime = backend.CudaRuntime(rank)
        runtime.init_comm(
            "sendrecv",
            world_size,
            rank,
        )

        rightpath = "rightsendrecv.onnx"
        rightmodel = onnx.load(rightpath)
        stub = OnnxStub(rightmodel, runtime)
        # input_data = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], dtype=np.float32).reshape(3,3)
        # next(iter(stub.inputs.values())).copyin_numpy(input_data)
        stub.run()
        print(rank, next(stub.outputs.values().__iter__()).copyout_numpy())
    else:
        runtime = backend.CudaRuntime(rank)
        runtime.init_comm(
            "sendrecv",
            world_size,
            rank,
        )


world_size = 3

workers = [
    mp.Process(
        target=start_workers,
        args=(rank, world_size),
    )
    for rank in range(world_size)
]

for w in workers:
    w.start()

for w in workers:
    w.join()
