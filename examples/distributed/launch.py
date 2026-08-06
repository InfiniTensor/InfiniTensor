import argparse
import multiprocessing as mp
import os
import time

import numpy as np
import onnx

from parallel_opt import parallel_model
from pyinfinitensor.onnx import OnnxStub, backend


def parse_args():
    parser = argparse.ArgumentParser(description="Launch distributed InfiniTensor")
    parser.add_argument("--device", required=True, help="InfiniRT device name")
    parser.add_argument("--num-nodes", type=int, default=1)
    parser.add_argument("--nproc-per-node", type=int, default=1)
    parser.add_argument("--name", default="test")
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--matmul-compute-type",
        choices=("default", "fp16", "tf32"),
        default="default",
    )
    return parser.parse_args()


def run_model(model, runtime, inputs, matmul_compute_type, repeats=10):
    stub = OnnxStub(model, runtime, matmul_compute_type=matmul_compute_type)
    for tensor, value in zip(stub.inputs.values(), inputs, strict=False):
        tensor.copyin_numpy(value)
    stub.run()
    output = next(iter(stub.outputs.values())).copyout_numpy()

    begin = time.time()
    for _ in range(repeats):
        stub.run()
    print(f"average time: {(time.time() - begin) / repeats}")
    return output


def run_and_compare(name, model, runtime, matmul_compute_type):
    input_ids = np.load(f"{name}_inputs.npy")
    position_ids = np.arange(input_ids.shape[-1])
    expected = np.load(f"{name}_results.npy")
    output = run_model(
        model, runtime, (input_ids, position_ids), matmul_compute_type
    )
    print("output absolute mean:", np.abs(output).mean())
    print("maximum absolute difference:", np.abs(output - expected).max())


def start_worker(
    device, name, world_size, rank, local_rank, model, matmul_compute_type
):
    distributed_name = f"{name}_dist"
    model = parallel_model(model, world_size, rank)
    external_path = f"./{distributed_name}_rank{rank}.pb"
    if os.path.exists(external_path):
        os.remove(external_path)
    onnx.save_model(
        model,
        f"./{distributed_name}_rank{rank}.onnx",
        save_as_external_data=True,
        location=external_path,
    )
    runtime = backend.runtime(device, local_rank)
    runtime.init_comm(distributed_name, world_size, rank)
    run_and_compare(name, model, runtime, matmul_compute_type)


def main():
    args = parse_args()
    model = onnx.load(args.model)
    world_size = args.num_nodes * args.nproc_per_node
    workers = [
        mp.Process(
            target=start_worker,
            args=(
                args.device,
                args.name,
                world_size,
                rank,
                rank % args.nproc_per_node,
                model,
                args.matmul_compute_type,
            ),
        )
        for rank in range(world_size)
    ]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join()
        if worker.exitcode != 0:
            raise RuntimeError(f"worker exited with status {worker.exitcode}")


if __name__ == "__main__":
    main()
