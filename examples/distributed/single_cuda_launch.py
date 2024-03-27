import argparse
import os
import time
import multiprocessing as mp
from pyinfinitensor.onnx import OnnxStub, backend
import onnx
from onnx.external_data_helper import convert_model_to_external_data
from onnx.shape_inference import infer_shapes_path
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
        "--name", type=str, default="test", help="name of this instance."
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
        args.name,
        args.model,
        args.batch_size,
        args.length,
        args.gen_std,
    )


def run_model(model, runtime, inputs, n=10):
    stub = OnnxStub(model, runtime)
    for tensor, input in zip(stub.inputs.values(), inputs, strict=False):
        tensor.copyin_numpy(input)
    # stub.tune()
    stub.run()
    # get outputs
    outputs = next(stub.outputs.values().__iter__()).copyout_numpy()

    # bench
    for tensor, input in zip(stub.inputs.values(), inputs, strict=False):
        tensor.copyin_numpy(input)
    begin = time.time()
    for _ in range(n):
        stub.run()
    end = time.time()
    avg_time = (end - begin) / n
    print(f"average time: {avg_time}")
    return outputs


def run_and_compare(name, model, runtime):
    input_ids = np.load(f"{name}_inputs.npy")
    position_ids = np.arange(input_ids.shape[-1])
    results = np.load(f"{name}_results.npy")
    outputs = run_model(model, runtime, (input_ids, position_ids))
    print("outputs abs mean:", abs(outputs).mean())
    print("max abs diff:", abs(outputs - results).max())


def start_worker(
    name: str, world_size: int, rank: int, local_rank: int, model: onnx.ModelProto
):
    dist_name = name + "_dist"
    model = parallel_model(model, world_size, rank)
    extern_path = f"./{dist_name}_rank{rank}.pb"
    if os.path.exists(extern_path):
        os.remove(extern_path)
    onnx.save_model(
        model,
        f"./{dist_name}_rank{rank}.onnx",
        save_as_external_data=True,
        location=extern_path,
    )
    #infer_shapes_path(f"./{dist_name}_rank{rank}.onnx")
    runtime = backend.CudaRuntime(local_rank)
    # print("init comm")
    runtime.init_comm(
        dist_name,
        world_size,
        rank,
    )
    run_and_compare(name, model, runtime)


def start_single(name, model):
    runtime = backend.CudaRuntime(0)
    run_and_compare(name, model, runtime)


def gen_standard(name, model, voc_size, bs, len):
    # generate standard results
    input_ids = np.random.randint(0, voc_size, (bs, len))
    position_ids = np.arange(len)
    np.save(f"{name}_inputs", input_ids)
    runtime = backend.CudaRuntime(0)
    outputs = run_model(model, runtime, (input_ids, position_ids), 1)
    print("outputs abs mean:", abs(outputs).mean())
    np.save(f"{name}_results", outputs)


def main():
    nnodes, nproc_per_node, name, model_path, bs, length, gen_std = parse_args()

    model = onnx.load(model_path)

    # generate standart output
    if gen_std:
        print(f"generate standard data for {name}.")
        # a small vocabulary size to fit all LLM.
        voc_size = 1000
        gen_standard(name, model, voc_size, bs, length)
        return

    # run single process.
    # use standalone process to isolate cuda.
    print("run model by single GPU.")
    start_single(name, model)


if __name__ == "__main__":
    main()