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
    parser.add_argument(
        "--type", type=str, choices=["fp32", "fp16", "tf32"], default="fp32", help="data type"
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
        args.type,
    )


def run_model(model, runtime, world_size=1, rank=0, n=10, data_type="default"):
    stub = OnnxStub(model, runtime, matmul_compute_type=data_type)
    load_inputs(stub, world_size, rank)
    # stub.tune()
    stub.run()
    # get outputs
    outputs = next(stub.outputs.values().__iter__()).copyout_numpy()

    # bench
    for _ in range(n):
        stub.run()
    begin = time.time()
    for _ in range(n * 2):
        stub.run()
    end = time.time()
    avg_time = (end - begin) / (n * 2)
    print(f"average time: {avg_time}")
    return outputs

def load_inputs(stub, world_size=1, rank=0):
    for i, (name, tensor) in enumerate(stub.inputs.items()):
        input = np.load(f"./data/input_{i}.npy")
        if all(x == y for x,y in zip(input.shape,tensor.shape())):
            tensor.copyin_numpy(input)
        else:
            tensor.copyin_numpy(np.hsplit(input, world_size)[rank])


def run_and_compare(name, model, runtime, world_size=1, rank=0, data_type="default"):
    results = np.load(f"./data/output.npy")
    outputs = run_model(model, runtime, world_size, rank, data_type=data_type)
    print("outputs abs mean:", abs(outputs).mean())
    print("max abs diff:", abs(outputs - results).max())

# def getDiff(base, test):
#     absolute_diff = np.abs(np.subtract(base, test))
#     max_absolute_diff = np.max(absolute_diff)

#     baseCopy = base.astype(np.float64).ravel()
#     testCopy = test.astype(np.float64).ravel()
#     upValue = np.sum(np.abs(baseCopy - testCopy))
#     downValue = np.sum(np.abs(baseCopy)) + np.float64(1e-9)
#     max_relative_diff = upValue / downValue
#     print(f"Max absolute difference: {max_absolute_diff}\n"
#           f"Max relative difference: {max_relative_diff}")
#     return max_absolute_diff, max_relative_diff

def start_worker(
    name: str, world_size: int, rank: int, local_rank: int, model: onnx.ModelProto, data_type: str
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
    runtime = backend.BangRuntime(local_rank)
    # print("init comm")
    runtime.init_comm(
        dist_name,
        world_size,
        rank,
    )
    run_and_compare(name, model, runtime, world_size, rank, data_type)


def start_single(name, model, data_type):
    runtime = backend.BangRuntime(0)
    run_and_compare(name, model, runtime, data_type=data_type)

def generate_input_output(model):
    os.makedirs(os.path.dirname("./data/"), exist_ok=True)
    runtime = backend.BangRuntime(0)
    stub = OnnxStub(model, runtime)
    position_id = 0
    for i, (name, tensor) in enumerate(stub.inputs.items()):
        input = tensor.copyout_numpy()
        if np.issubdtype(input.dtype, np.integer):
            if input.size == 1:
                # input = np.array([position_id])
                input = np.random.randint(0,2,size=input.shape, dtype=input.dtype)
            else:
                input = np.random.randint(0,2,size=input.shape, dtype=input.dtype)
        elif input.dtype == np.bool_:
            input = np.random.randint(0,2,size=input.shape) > 0
        else:
            if i == 0:
                input = np.ones(input.shape).astype(input.dtype)
                position_id = input.shape[-1] - 1
            else:
                input = np.random.rand(*input.shape).astype(input.dtype)
        tensor.copyin_numpy(input)
        np.save(f"./data/input_{i}", input)
    stub.run()
    time.sleep(0.01)
    output = next(stub.outputs.values().__iter__()).copyout_numpy()
    if np.isnan(output).any():
        print("Nan in output")
    np.save(f"./data/output", output)


def main():
    nnodes, nproc_per_node, name, model_path, bs, length, gen_std, data_type = parse_args()
    data_type = "default" if data_type == "fp32" else data_type
    
    model = onnx.load(model_path)

    # generate standart output
    if gen_std:
        print(f"generate standard data for {name}.")
        # a small vocabulary size to fit all LLM.
        generate_input_output(model)
        return

    if nproc_per_node == 1:
        # run single process.
        # use standalone process to isolate bang.
        print("run model by single MLU.")
        # p = mp.Process(target=start_single, args=(name, model, data_type))
        # p.start()
        # p.join()
        start_single(name, model, data_type)
        return

    # run distributed parallel.
    world_size = nnodes * nproc_per_node
    print(f"run model by {world_size} MLU in parallel.")
    workers = [
        mp.Process(
            target=start_worker,
            args=(name, world_size, rank, rank % nproc_per_node, model, data_type),
        )
        for rank in range(world_size)
    ]

    for w in workers:
        w.start()

    for w in workers:
        w.join()


if __name__ == "__main__":
    main()
