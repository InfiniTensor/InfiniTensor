import argparse
import os
import time
import multiprocessing as mp
from pyinfinitensor.onnx import OnnxStub, backend
import onnx
from onnx.external_data_helper import convert_model_to_external_data
import numpy as np
from parallel_opt import parallel_model


os.environ["NVIDIA_TF32_OVERRIDE"] = "0"


def parse_args():
    parser = argparse.ArgumentParser(description="launch distributed infinitensor")
    parser.add_argument("--num_nodes", type=int, default=1, help="number of nodes")
    parser.add_argument(
        "--nproc_per_node", type=int, default=1, help="number of processes per node"
    )
    parser.add_argument(
        "--model", type=str, required=True, help="path to the ONNX model file."
    )
    parser.add_argument("--batch_size", type=int, default=1, help="batch size.")
    parser.add_argument("--length", type=int, default=1, help="sequence length.")
    parser.add_argument(
        "--gen_std",
        action="store_true",
        help="whether to generate the standard results.",
    )
    args = parser.parse_args()
    print("arg setting: ", args)
    return (
        args.num_nodes,
        args.nproc_per_node,
        args.model,
        args.batch_size,
        args.length,
        args.gen_std,
    )


def run_stub(stub: OnnxStub, inputs: np.array, n=100):
    # warm up
    next(stub.inputs.items().__iter__())[1].copyin_int32(inputs.reshape(-1).tolist())
    stub.tune()
    for _ in range(20):
        stub.run()
    outputs = np.array(next(stub.outputs.items().__iter__())[1].copyout_float())

    # bench
    next(stub.inputs.items().__iter__())[1].copyin_int32(inputs.reshape(-1).tolist())
    begin = time.time()
    for _ in range(n):
        stub.run()
    end = time.time()
    outputs = np.array(next(stub.outputs.items().__iter__())[1].copyout_float())
    avg_time = (end - begin) / n
    print(f"average time: {avg_time}")
    return outputs


def start_worker(
    dist_name: str, world_size: int, rank: int, local_rank: int, model: onnx.ModelProto
):
    print("start worker")
    model = parallel_model(model, world_size, rank)
    extern_path = f"./dist_model_rank{rank}.pb"
    if os.path.exists(extern_path):
        os.remove(extern_path)
    convert_model_to_external_data(
        model,
        all_tensors_to_one_file=True,
        location=extern_path,
        size_threshold=1024,
        convert_attribute=False,
    )
    onnx.save(model, f"dist_model_rank{rank}.onnx")
    runtime = backend.CudaRuntime(local_rank)
    print("init comm")
    runtime.init_comm(
        dist_name,
        world_size,
        rank,
    )
    print("load model")
    stub = OnnxStub(model, runtime)
    data = np.load("inputs.npy")
    print("run model")
    outputs = run_stub(stub, data)
    print("outputs sum:", outputs.sum())
    results = np.load("results.npy")
    print("max abs diff:", abs(outputs - results).max())
    print("max rel diff:", abs((outputs - results) / results).max())
    # assert np.allclose(outputs, results, rtol=1e-3, atol=1e-6)


def run_standard(model, voc_size=50272, bs=1, len=2048):
    # generate standard results
    runtime = backend.CudaRuntime(0)
    stub = OnnxStub(model, runtime)
    # data = np.zeros((bs, len), dtype=np.int32)
    data = np.random.randint(0, voc_size, (bs, len), dtype=np.int32)
    np.save("inputs", data)
    outputs = run_stub(stub, data)
    print("outputs sum:", outputs.sum())
    np.save("results", outputs)


def main():
    nnodes, nproc_per_node, model_path, bs, length, gen_std = parse_args()

    model = onnx.load(model_path)

    if gen_std:
        p = mp.Process(target=run_standard, args=(model, bs, length))
        p.start()
        p.join()

    world_size = nnodes * nproc_per_node
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
