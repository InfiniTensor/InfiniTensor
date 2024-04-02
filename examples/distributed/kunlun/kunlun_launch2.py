import sys
sys.path.append("../")
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

st_input_dir = ".cache/input/"
st_output_dir = ".cache/output/"

def parse_args():
    parser = argparse.ArgumentParser(description="launch distributed infinitensor")
    parser.add_argument("--num_nodes", type=int, default=1, help="number of nodes")
    parser.add_argument(
        "--nproc_per_node", type=int, default=2, help="number of processes per node"
    )
    parser.add_argument(
        "--name", type=str, default="test", help="name of this instance."
    )
    parser.add_argument(
        "--model", type=str, default="/data1/shared/panzezhong/llama/fp32/my_llama_fp32.sim.onnx", help="path to the ONNX model file."
    )
    parser.add_argument("--batch_size", type=int, default=1, help="batch size.")
    parser.add_argument("--length", type=int, default=1, help="sequence length.")
    parser.add_argument(
        "--gen_std",
        default=False,
        action="store_true",
        help="whether to generate the standard results.",
    )
    parser.add_argument(
        "--run_single",
        default=False,
        action="store_true",
        help="whether run model with single process with standard inputs"
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
        args.run_single
    )


def run_model(model, runtime, world_size=1, rank=0, n=10):
    stub = OnnxStub(model, runtime)
    load_inputs(stub, world_size, rank)
    # stub.tune()
    stub.run()
    # get outputs
    time.sleep(0.01)
    outputs = next(stub.outputs.values().__iter__()).copyout_numpy()

    # bench
    begin = time.time()
    for _ in range(n):
        stub.run()
    end = time.time()
    avg_time = (end - begin) / n
    print(f"average time: {avg_time}")
    return outputs



def run_and_compare(name, model, runtime, world_size=1, rank = 0):
    results = np.load(os.path.join(st_output_dir, "test_output.npy"))
    outputs = run_model(model, runtime, world_size, rank)
    print(outputs[:100])
    if np.isnan(outputs).any():
        print("Nan in output")
    print("answer argmax:", np.argmax(results))
    print("output argmax:", np.argmax(outputs))
    #np.testing.assert_allclose(outputs, results, rtol=1e-3, atol=1e-3)
    getDiff(results, outputs)


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
    infer_shapes_path(f"./{dist_name}_rank{rank}.onnx")
    runtime = backend.KUNLUNRuntime(local_rank)
    # print("init comm")
    runtime.init_comm(
        dist_name,
        world_size,
        rank,
    )
    run_and_compare(name, model, runtime, world_size, rank)


def start_single(name, model):
    runtime = backend.KUNLUNRuntime(0)
    run_and_compare(name, model, runtime)


def generate_input_output(model):
    runtime = backend.KUNLUNRuntime(0)
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
        np.save(os.path.join(st_input_dir, f"input_{i}"), input)
    stub.run()
    # print(stub.outputs)
    time.sleep(0.01)
    output = next(stub.outputs.values().__iter__()).copyout_numpy()
    print(output[:100])
    if np.isnan(output).any():
        print("Nan in output")
    np.save(os.path.join(st_output_dir, f"output"), output)


def load_inputs(stub, world_size=1, rank=0):
    for i, (name, tensor) in enumerate(stub.inputs.items()):
        input = np.load(os.path.join(st_input_dir, f"test_input_{name}.npy"))
        if all(x == y for x,y in zip(input.shape,tensor.shape())):
            tensor.copyin_numpy(input)
        else:
            tensor.copyin_numpy(np.hsplit(input, world_size)[rank])


def getDiff(base, test):
    absolute_diff = np.abs(np.subtract(base, test))
    max_absolute_diff = np.max(absolute_diff)

    baseCopy = base.astype(np.float64).ravel()
    testCopy = test.astype(np.float64).ravel()
    upValue = np.sum(np.abs(baseCopy - testCopy))
    downValue = np.sum(np.abs(baseCopy)) + np.float64(1e-9)
    max_relative_diff = upValue / downValue
    print(f"Max absolute difference: {max_absolute_diff}\nMax relative difference: {max_relative_diff}")

    return max_absolute_diff, max_relative_diff


def main():
    nnodes, nproc_per_node, name, model_path, bs, length, gen_std, run_single = parse_args()

    model = onnx.load(model_path)

    # generate standart output
    if gen_std:
        print("Generate inputs and outputs.")
        p = mp.Process(target=generate_input_output, args=[model])
        p.start()
        p.join()
        return

    # # run single process.
    # # use standalone process to isolate cuda.
    if run_single:
        print("run model by single GPU.")
        p = mp.Process(target=start_single, args=(name, model))
        p.start()
        p.join()
        return 

    # run distributed parallel.
    world_size = nnodes * nproc_per_node
    print(f"run model by {world_size} GPU in parallel.")
    workers = [
        mp.Process(
            target=start_worker,
            args=(name, world_size, rank, rank % nproc_per_node, model),
        )
        for rank in range(world_size)
    ]

    for w in workers:
        w.start()

    for w in workers:
        w.join()


if __name__ == "__main__":
    main()