import argparse
import os
import time
import multiprocessing as mp
from pyinfinitensor.onnx import OnnxStub, backend
import onnx
import numpy as np
from parallel import parallel_model


def parse_args():
    parser = argparse.ArgumentParser(description="launch distributed infinitensor")
    parser.add_argument("--num_nodes", type=int, default=1, help="number of nodes")
    parser.add_argument(
        "--nproc_per_node", type=int, default=1, help="number of processes per node"
    )
    parser.add_argument(
        "--model", type=str, required=True, help="path to the ONNX model file."
    )
    args = parser.parse_args()
    print("arg setting: ", args)
    return args.num_nodes, args.nproc_per_node, args.model


def run_stub(stub: OnnxStub, inputs: np.array, n=100):
    stub.init()
    # warm up
    next(stub.inputs.items().__iter__())[1].copyin_float(inputs.reshape(-1).tolist())
    stub.tune()
    for _ in range(20):
        stub.run()
    outputs = np.array(next(stub.outputs.items().__iter__())[1].copyout_float())

    # bench
    next(stub.inputs.items().__iter__())[1].copyin_float(inputs.reshape(-1).tolist())
    begin = time.time()
    for _ in range(n):
        stub.run()
    end = time.time()
    outputs = np.array(next(stub.outputs.items().__iter__())[1].copyout_float())
    print(outputs.shape)

    avg_time = (end - begin) / n
    return avg_time


def start_worker(
    dist_name: str, world_size: int, rank: int, local_rank: int, model: onnx.ModelProto
):
    print("start worker")
    runtime = backend.CudaRuntime(local_rank)
    print("init comm")
    runtime.init_comm(
        dist_name,
        world_size,
        rank,
    )
    model = parallel_model(model, world_size, rank)
    onnx.save(model, f"dist_model_rank{rank}.onnx")
    print("load model")
    stub = OnnxStub(model, runtime)
    data = np.random.randn(1, 3, 224, 224)
    print("run model")
    avg_time = run_stub(stub, data)
    print(f"average time: {avg_time}")


def main():
    nnodes, nproc_per_node, model_path = parse_args()
    world_size = nnodes * nproc_per_node

    model = onnx.load(model_path)

    dist_name = f"dist_{os.getpid()}"
    workers = [
        mp.Process(
            target=start_worker,
            args=(dist_name, world_size, rank, rank % nproc_per_node, model),
        )
        for rank in range(world_size)
    ]

    for w in workers:
        w.start()

    for w in workers:
        w.join()


if __name__ == "__main__":
    main()
