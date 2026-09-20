"""Two-process CUDA TP smoke test for a dynamic-batch Gemm model.

This is an integration test for the existing examples/distributed/parallel.py
path. It intentionally uses a tiny model so it can run on a shared server.
"""

import copy
import multiprocessing as mp
import os
import sys

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from examples.distributed.parallel import parallel_model
from pyinfinitensor.onnx import OnnxStub, backend


WORLD_SIZE = 2
WORKSPACE_SIZE = 64 << 20


def make_model(trans_b=False):
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [None, 4])
    position = helper.make_tensor_value_info("position", TensorProto.FLOAT, [4])
    hidden = helper.make_tensor_value_info("hidden", TensorProto.FLOAT, [None, 4])
    output = helper.make_tensor_value_info("output", TensorProto.FLOAT, [None, 4])
    weight = numpy_helper.from_array(np.eye(4, dtype=np.float32), name="weight")
    bias = numpy_helper.from_array(np.zeros(4, dtype=np.float32), name="bias")
    nodes = [
        helper.make_node("Add", ["x", "position"], ["hidden"]),
        helper.make_node(
            "Gemm", ["hidden", "weight", "bias"], ["output"], transB=int(trans_b)
        ),
    ]
    graph = helper.make_graph(nodes, "dynamic_tp_smoke", [x, position], [output], [weight, bias])
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])


def inputs(batch):
    x = np.arange(batch * 4, dtype=np.float32).reshape(batch, 4)
    position = np.arange(4, dtype=np.float32)
    return x, position


def run_single(model, batch):
    runtime = backend.cuda_runtime(workspace_size=WORKSPACE_SIZE)
    stub = OnnxStub(model, runtime)
    x, position = inputs(batch)
    stub.set_input([[batch, 4], [4]])
    stub.inputs["x"].copyin_numpy(x)
    stub.inputs["position"].copyin_numpy(position)
    stub.run()
    return np.asarray(stub.outputs["output"].copyout_float()).reshape(batch, 4)


def run_rank(rank, model_bytes, batches):
    model = onnx.load_from_string(model_bytes)
    model = parallel_model(model, WORLD_SIZE, rank)
    runtime = backend.cuda_runtime(rank, workspace_size=WORKSPACE_SIZE)
    stub = OnnxStub(model, runtime)
    runtime.init_comm("dynamic_tp_smoke", WORLD_SIZE, rank)
    results = {}
    for batch in batches:
        x, position = inputs(batch)
        stub.set_input([[batch, 4], [4]])
        stub.inputs["x"].copyin_numpy(x)
        stub.inputs["position"].copyin_numpy(position)
        stub.run()
        results[str(batch)] = np.asarray(
            stub.outputs["output"].copyout_float()
        ).reshape(batch, 4)
    np.savez(f"dynamic_tp_smoke_rank{rank}.npz", **results)


def main():
    batches = (1, 5, 2, 8, 1)
    ctx = mp.get_context("spawn")
    for variant, trans_b in (("normal", False), ("transposed_b", True)):
        model = make_model(trans_b=trans_b)
        baseline = {
            batch: run_single(copy.deepcopy(model), batch) for batch in batches
        }
        model_bytes = model.SerializeToString()
        workers = [
            ctx.Process(target=run_rank, args=(rank, model_bytes, batches))
            for rank in range(WORLD_SIZE)
        ]
        for worker in workers:
            worker.start()
        for worker in workers:
            worker.join()
            if worker.exitcode != 0:
                raise SystemExit(
                    f"{variant} worker failed with exit code {worker.exitcode}"
                )
        for rank in range(WORLD_SIZE):
            results = np.load(f"dynamic_tp_smoke_rank{rank}.npz")
            for batch in batches:
                np.testing.assert_allclose(
                    results[str(batch)], baseline[batch], rtol=1e-5, atol=1e-5
                )
    print("dynamic TP smoke: 2 processes, 2 GPUs, 5 shapes, normal/transposed Gemm allclose PASS")


if __name__ == "__main__":
    main()
