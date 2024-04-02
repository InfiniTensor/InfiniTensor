import sys
sys.path.append('../')

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
from functools import wraps


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
        "--model", type=str, default="", help="path to the ONNX model file."
    )
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
    parser.add_argument(
        "--input_dir",
        default="./",
        help="path to save model input data"
    )
    parser.add_argument(
        "--result_dir",
        default="./",
        help="path to save model standard output"
    )
    parser.add_argument(
        "--internal_model_dir",
        default="./",
        help="path to save internal onnx model for parallel run"
    )
    args = parser.parse_args()
    
    # check path, mkdir if not exist
    check_exists(args.input_dir)
    check_exists(args.result_dir)
    check_exists(args.internal_model_dir)
    
    print("arg setting: ", args)
    return (
        args.num_nodes,
        args.nproc_per_node,
        args.name,
        args.model,
        args.gen_std,
        args.run_single,
        args.input_dir,
        args.result_dir,
        args.internal_model_dir
    )


"""
utils function for this scripts
"""
def check_exists(path: str):
    if not os.path.exists(path):
        os.makedirs(path)

def np_assert(base, test, rtol=1e-2, atol=1e-1):
    # np.testing.assert_allclose(test, base, rtol, atol)
    print("max abs diff:", abs(base - test).max())


"""
Perf wrapper, run function n times
then average
"""
def perf_it(n):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # warmup
            for _ in range(n):
                func(*args, **kwargs)
            
            t_total = 0
            for _ in range(n):
                t0 = time.time()
                func(*args, **kwargs)
                t1 = time.time()
                t_total += t1 - t0
            avg_time = (t_total) / n
            print(f"Avg runtime of {n} time is {avg_time:.6f} seconds")
            return avg_time
        return wrapper
    return decorator


"""
Run InfiniTensor model with Standard input 
check=True: check with standard output gen by pytorch
perf=True: run n times to get avg time
"""
def run_model(task_name,
              model, 
              runtime, 
              world_size=1, 
              rank=0, 
              n=10,
              check=True,
              perf=True):
    
    stub = OnnxStub(model, runtime)

    # load in Onnx model inputs
    def load_inputs(stub: OnnxStub):
        # check exists
        inputs = []
        for i, (name, tensor) in enumerate(stub.inputs.items()):
            input_path = os.path.join(input_dir, \
                                f"{task_name}_input_{i}.npy")
            print(input_path)
            if os.path.exists(input_path):
                input = np.load(input_path)
            else :
                raise KeyError(f"{i} th input of model not exists")
            # check shape
            if all(x == y for x,y in zip(input.shape, tensor.shape())):
                tensor.copyin_numpy(input)
            else:
                tensor.copyin_numpy(np.hsplit(input, world_size)[rank])

    load_inputs(stub)
    # stub.tune()
    stub.run()
    time.sleep(0.01)
    output = next(stub.outputs.values().__iter__()).copyout_numpy()

    # check output results with standard output
    if check:
        st_output_path = os.path.join(result_dir, \
                                f"{task_name}_output.npy")
        assert os.path.exists(st_output_path) , \
                    "standard output not exists"
        st_output = np.load(st_output_path)
        if np.isnan(output).any():
            print("Nan in output")
            exit()
        np_assert(st_output, output)        

    # perf 
    if perf:
        @perf_it(n)
        def perf_infinitensor(stub: OnnxStub):
            stub.run()
        perf_infinitensor(stub)

    return output


"""
Start a worker in Parallel
"""
def start_worker(name: str, 
           world_size: int, 
           rank: int, 
           local_rank: int, 
           model: onnx.ModelProto):

    dist_name = name + "_dist"
    # partial a onnx model to world_size part
    model = parallel_model(model, world_size, rank)
    onnx.save(model, os.path.join(internal_model_dir, \
                                    f"{dist_name}_rank{rank}.onnx"))
    runtime = backend.KUNLUNRuntime(local_rank)
    # print("init comm")
    runtime.init_comm(
        dist_name,
        world_size,
        rank,
    )
    run_model(name, model, runtime, world_size, rank)


"""
generate standard input/output with 
sigle card run
"""
def gen_stardard(task_name: str, model: onnx.ModelProto):
    runtime = backend.KUNLUNRuntime(0)
    stub = OnnxStub(model, runtime)
    position_id = 0
    # generate random input for model
    for i, (name, tensor) in enumerate(stub.inputs.items()):
        input = tensor.copyout_numpy()
        if np.issubdtype(input.dtype, np.integer):
            if input.size == 1:
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
        np.save(os.path.join(input_dir, \
                    f"{task_name}_input_{i}.npy"), input)
    stub.run()
    # print(stub.outputs)
    output = next(stub.outputs.values().__iter__()).copyout_numpy()
    if np.isnan(output).any():
        print("Nan in output")
        exit()
    np.save(os.path.join(result_dir, f"{task_name}_output.npy"), output)


def main():

    global input_dir, result_dir, internal_model_dir
    
    nnodes, nproc_per_node, task_name, \
        model_path, gen_std, run_single, \
            input_dir, result_dir, internal_model_dir = parse_args()
    
    # load input onnx model
    model = onnx.load(model_path)

    # generate standart output
    if gen_std:
        print("Generate inputs and outputs.")
        gen_stardard(task_name, model)
        return

    if run_single:
        print("Run model by one GPU card.")
        runtime = backend.KUNLUNRuntime(0)
        run_model(task_name, model, runtime)
        return 

    # run distributed parallel.
    world_size = nnodes * nproc_per_node
    print(f"Run model by {world_size} GPU in parallel.")
    workers = [
        mp.Process(
            target=start_worker,
            args=(task_name, world_size, rank, rank % nproc_per_node, model),
        )
        for rank in range(world_size)
    ]

    for w in workers:
        w.start()

    for w in workers:
        w.join()


if __name__ == "__main__":
    main()
