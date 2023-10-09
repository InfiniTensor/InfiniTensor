import argparse
import os
import time
import multiprocessing as mp
from pyinfinitensor.onnx import OnnxStub, backend
import onnx
from onnx.external_data_helper import convert_model_to_external_data
import numpy as np
from parallel_opt import parallel_model
from tqdm import tqdm


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
        "--model1", type=str, required=True, help="path to the ONNX model file."
    )
    parser.add_argument(
        "--model2", type=str, required=True, help="path to the ONNX model file."
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
        args.model1,
        args.model2,
        args.batch_size,
        args.length,
        args.gen_std,
    )


def run_model(
    model1,
    model2,
    runtime1,
    runtime2,
    inputs1: np.array,
    inputs2: np.array,
    world_size=1,
    n=20,
):
    batchsize = 1
    size = 1
    ####################################
    # run the first graph without kvcache
    ####################################
    stub1 = OnnxStub(model1, runtime1)
    stub1.inputs["onnx::Reshape_0"].copyin_int32(inputs1.reshape(-1).tolist())
    stub1.tune()
    stub1.run()
    kvcache_it = []
    count = 0
    for output in stub1.outputs.items().__iter__():
        if count == 0:
            logits_it = np.array(output[1].copyout_float(), dtype=np.float32)
        else:
            kvcache_it.append(np.array(output[1].copyout_float(), dtype=np.float32))
        count = count + 1

    # # bench for stub1
    # next(stub1.inputs.items().__iter__())[1].copyin_int32(inputs1.reshape(-1).tolist())
    # begin = time.time()
    # for _ in range(n):
    #     stub1.run()
    # end = time.time()
    # avg_time = (end - begin) / n
    # print(f"stub1 average time: {avg_time}")

    ####################################
    # run the second graph with kvcache
    ####################################
    stub2 = OnnxStub(model2, runtime2)
    for i in tqdm(range(1, 200)):
        # input2 需要随机生成？
        input_shapes = []
        input_shapes.append([batchsize, size])
        for j in range(24):
            input_shapes.append([batchsize, 12 // world_size, i, 64])
        input_shapes.append([batchsize, 1])
        stub2.set_input(input_shapes)

        past_kvcache_length = (i + 2) * np.ones((batchsize, 1), dtype=np.int32)
        # copyin input
        stub2.inputs["onnx::Reshape_0"].copyin_int32(inputs2.reshape(-1).tolist())
        stub2.inputs["input.3"].copyin_int32(past_kvcache_length.reshape(-1).tolist())
        count = -1
        for input in stub2.inputs.items().__iter__():
            if count in range(24):
                input[1].copyin_float(kvcache_it[count].reshape(-1).tolist())
            count = count + 1
        # stub2.tune()
        stub2.run()

        # copyout output
        count = 0
        kvcache_it = []
        for output in stub2.outputs.items().__iter__():
            if count == 0:
                logits_it = np.array(output[1].copyout_float())
            else:
                kvcache_it.append(np.array(output[1].copyout_float()))
            count = count + 1

    # # bench for stub2
    # # copyin input
    # stub2.inputs['onnx::Reshape_0'].copyin_int32(inputs2.reshape(-1).tolist())
    # stub2.inputs['input.3'].copyin_int32(past_kvcache_length.reshape(-1).tolist())
    # count = -1
    # for input in stub2.inputs.items().__iter__():
    #     if count in range(24):
    #         input[1].copyin_float(kvcache_it[count].reshape(-1).tolist())
    #     count = count + 1
    # begin = time.time()
    # for _ in range(n):
    #     stub2.run()
    # end = time.time()
    # avg_time = (end - begin) / n
    # print(f"stub2 average time: {avg_time}")
    return logits_it


def run_and_compare(name, model1, model2, runtime1, runtime2, world_size=1):
    data1 = np.load(f"{name}_inputs1.npy")
    data2 = np.load(f"{name}_inputs2.npy")
    results = np.load(f"{name}_results.npy")
    outputs = run_model(model1, model2, runtime1, runtime2, data1, data2, world_size)
    print("outputs sum:", outputs.sum())
    print("max abs diff:", abs(outputs - results).max())
    print("max rel diff:", abs((outputs - results) / results).max())
    # assert np.allclose(outputs, results, rtol=1e-3, atol=1e-6)


def start_worker(
    name: str,
    world_size: int,
    rank: int,
    local_rank: int,
    model1: onnx.ModelProto,
    model2: onnx.ModelProto,
):
    dist_name = name + "_dist"
    ####################################
    # shard the first graph
    ####################################
    model1 = parallel_model(model1, world_size, rank)
    extern_path = f"./{dist_name}_stub1_rank{rank}.pb"
    if os.path.exists(extern_path):
        os.remove(extern_path)
    convert_model_to_external_data(
        model1,
        all_tensors_to_one_file=True,
        location=extern_path,
        size_threshold=1024,
        convert_attribute=False,
    )
    onnx.save(model1, f"./{dist_name}_stub1_rank{rank}.onnx")
    runtime1 = backend.CudaRuntime(local_rank)
    runtime1.init_comm(
        dist_name,
        world_size,
        rank,
    )

    ####################################
    # shard the second graph
    ####################################
    model2 = parallel_model(model2, world_size, rank)
    extern_path = f"./{dist_name}_stub2_rank{rank}.pb"
    if os.path.exists(extern_path):
        os.remove(extern_path)
    convert_model_to_external_data(
        model2,
        all_tensors_to_one_file=True,
        location=extern_path,
        size_threshold=1024,
        convert_attribute=False,
    )
    onnx.save(model2, f"./{dist_name}_stub2_rank{rank}.onnx")
    runtime2 = backend.CudaRuntime(local_rank)
    # print("init comm")
    runtime2.init_comm(
        dist_name,
        world_size,
        rank,
    )

    # run the two graphs
    run_and_compare(name, model1, model2, runtime1, runtime2, world_size)


def start_single(name, model1, model2):
    runtime1 = backend.CudaRuntime(0)
    runtime2 = backend.CudaRuntime(0)
    run_and_compare(name, model1, model2, runtime1, runtime2)


def gen_standard(name, model1, model2, voc_size, bs, len):
    # generate standard results
    data1 = np.random.randint(0, voc_size, (bs, len), dtype=np.int32)
    data2 = np.random.randint(0, voc_size, (bs, len), dtype=np.int32)
    np.save(f"{name}_inputs1", data1)
    np.save(f"{name}_inputs2", data2)
    runtime1 = backend.CudaRuntime(0)
    runtime2 = backend.CudaRuntime(0)
    outputs = run_model(model1, model2, runtime1, runtime2, data1, data2, n=1)
    np.save(f"{name}_results", outputs)


def main():
    (
        nnodes,
        nproc_per_node,
        name,
        model1_path,
        model2_path,
        bs,
        length,
        gen_std,
    ) = parse_args()

    model1 = onnx.load(model1_path)
    model2 = onnx.load(model2_path)

    # generate standart output
    if gen_std:
        print(f"generate standard data for {name}.")
        # a small vocabulary size to fit all LLM.
        voc_size = 1000
        gen_standard(name, model1, model2, voc_size, bs, length)
        return

    # run single process.
    # use standalone process to isolate cuda.
    p = mp.Process(target=start_single, args=(name, model1, model2))
    p.start()
    p.join()

    # run distributed parallel.
    world_size = nnodes * nproc_per_node
    workers = [
        mp.Process(
            target=start_worker,
            args=(name, world_size, rank, rank % nproc_per_node, model1, model2),
        )
        for rank in range(world_size)
    ]

    for w in workers:
        w.start()

    for w in workers:
        w.join()


if __name__ == "__main__":
    main()
