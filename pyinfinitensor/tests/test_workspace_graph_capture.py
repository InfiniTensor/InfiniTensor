import os
import unittest

import numpy as np
from onnx import TensorProto, helper, numpy_helper

from pyinfinitensor import backend
from pyinfinitensor.onnx import OnnxStub
from pyinfinitensor.embedding import fp32_inference_policy


@unittest.skipUnless(hasattr(backend, "InfiniRuntime"), "Infini backend not built")
class TestWorkspaceGraphCapture(unittest.TestCase):
    def setUp(self):
        runtime = backend.runtime(backend.default_infini_device())
        handler = backend.GraphHandler(runtime)
        handler.data_malloc()
        try:
            handler.run_with_graph()
        except RuntimeError as error:
            if os.environ.get("INFINITENSOR_REQUIRE_GRAPH_CAPTURE"):
                raise
            self.skipTest(f"Graph API unavailable: {error}")

    def test_batch_norm_and_max_pool(self):
        if not backend.has_infini_aten_kernels:
            self.skipTest("BN/MaxPool require USE_INFINIOPS_ATEN_KERNELS")
        for kind in ("bn", "max", "bn_max"):
            for dtype, proto in ((np.float32, TensorProto.FLOAT),
                                 (np.float16, TensorProto.FLOAT16)):
                with self.subTest(kind=kind, dtype=dtype):
                    runtime = backend.runtime(backend.default_infini_device())
                    handler = backend.GraphHandler(runtime)
                    x = handler.tensor([2, 3, 4, 6], proto)
                    x.set_input()
                    out = x
                    parameters = []
                    if kind != "max":
                        for _ in range(4):
                            tensor = handler.tensor([3], TensorProto.FLOAT)
                            tensor.set_input()
                            parameters.append(tensor)
                        out = handler.batchNormalization(
                            out, None, *parameters, 0.9, 1e-5, False)
                    if kind != "bn":
                        out = handler.maxPool(out, None, 2, 2, 1, 1, 0, 0, 2, 2, 0)
                    out.set_output()
                    handler.data_malloc()
                    for iteration in range(6):
                        if iteration == 4:
                            runtime.clear_graph_cache()
                            self.assertEqual(runtime.graph_cache_size(), 0)
                        values = ((np.arange(144) * 17 % 41 - 20) / 4
                                  + iteration).astype(dtype).reshape(2, 3, 4, 6)
                        x.copyin_numpy(values)
                        expected = values.astype(np.float32)
                        if parameters:
                            arrays = [np.array(v, dtype=np.float32) for v in (
                                [iteration, -1, 2], [1, 2 + iteration, 3],
                                [1, -0.5, 2], [0.5, iteration, -2])]
                            for tensor, array in zip(parameters, arrays):
                                tensor.copyin_numpy(array)
                            mean, var, scale, bias = [a.astype(np.float32).reshape(1, 3, 1, 1)
                                                      for a in arrays]
                            expected = ((expected - mean) / np.sqrt(var + 1e-5)
                                        * scale + bias).astype(dtype).astype(np.float32)
                        if kind != "bn":
                            expected = expected.reshape(2, 3, 2, 2, 3, 2).max(axis=(3, 5))
                        if iteration == 0:
                            handler.run()
                        else:
                            handler.run_with_graph()
                            self.assertEqual(runtime.graph_capture_count(),
                                             1 if iteration < 4 else 2)
                            self.assertEqual(runtime.graph_cache_size(), 1)
                        np.testing.assert_allclose(out.copyout_numpy(), expected,
                                                   rtol=3e-3 if dtype == np.float16 else 1e-5,
                                                   atol=3e-3 if dtype == np.float16 else 1e-5)

    def test_layer_norm_workspace_survives_replay_and_cache_clear(self):
        runtime = backend.runtime(backend.default_infini_device())
        handler = backend.GraphHandler(runtime)
        x = handler.tensor([2, 3, 4], TensorProto.FLOAT)
        scale = handler.tensor([4], TensorProto.FLOAT)
        bias = handler.tensor([4], TensorProto.FLOAT)
        for tensor in (x, scale, bias):
            tensor.set_input()
        out = handler.layerNormalization(x, scale, None, bias, 1e-5, -1, 1)
        out.set_output()
        handler.data_malloc()
        for iteration in range(8):
            if iteration == 4:
                handler.clear_graph_cache()
                self.assertEqual(runtime.graph_cache_size(), 0)
            values = np.random.default_rng(iteration).normal(size=(2, 3, 4)).astype(np.float32)
            weight = np.array([1, .5, -1, 2], np.float32)
            offset = np.full(4, iteration / 10, np.float32)
            for tensor, value in ((x, values), (scale, weight), (bias, offset)):
                tensor.copyin_numpy(value)
            expected = ((values - values.mean(axis=-1, keepdims=True)) /
                        np.sqrt(values.var(axis=-1, keepdims=True) + 1e-5)) * weight + offset
            handler.run_with_graph()
            self.assertEqual(runtime.graph_capture_count(), 1 if iteration < 4 else 2)
            np.testing.assert_allclose(out.copyout_numpy(), expected, atol=1e-4, rtol=1e-4)

    def test_clearing_one_model_preserves_other_model_capture(self):
        runtime = backend.runtime(backend.default_infini_device(), 0, graph_cache_capacity=4)
        graphs = []
        for index in range(2):
            handler = backend.GraphHandler(runtime)
            x = handler.tensor([2, 3], TensorProto.FLOAT)
            x.set_input()
            y = handler.identity(x, None)
            y.set_output()
            handler.data_malloc()
            x.copyin_numpy(np.full((2, 3), index + 1, np.float32))
            handler.run_with_graph()
            graphs.append((handler, x, y))
        self.assertEqual(runtime.graph_cache_size(), 2)
        graphs[0][0].clear_graph_cache()
        self.assertEqual(runtime.graph_cache_size(), 1)
        values = np.arange(6, dtype=np.float32).reshape(2, 3)
        graphs[1][1].copyin_numpy(values)
        graphs[1][0].run_with_graph()
        self.assertEqual(runtime.graph_capture_count(), 2)
        np.testing.assert_array_equal(graphs[1][2].copyout_numpy(), values)

    def test_releasing_encoder_pool_preserves_other_encoder_replays(self):
        runtime = backend.runtime(backend.default_infini_device(), 0, graph_cache_capacity=4)
        weight = np.random.default_rng(7).normal(size=(32, 32)).astype(np.float32)
        model = helper.make_model(helper.make_graph([
            helper.make_node("MatMul", ["x", "w"], ["mm"]),
            helper.make_node("LayerNormalization", ["mm", "scale", "bias"], ["y"],
                             axis=-1, epsilon=1e-5),
        ], "encoder-pool-isolation", [helper.make_tensor_value_info("x", TensorProto.FLOAT, ["B", 17, 32])],
            [helper.make_tensor_value_info("y", TensorProto.FLOAT, ["B", 17, 32])],
            [numpy_helper.from_array(weight, "w"),
             numpy_helper.from_array(np.ones(32, np.float32), "scale"),
             numpy_helper.from_array(np.zeros(32, np.float32), "bias")]),
            opset_imports=[helper.make_opsetid("", 17)])
        models = [OnnxStub(model, runtime, execution_mode="graph") for _ in range(2)]

        def check(stub, batch, seed):
            values = np.random.default_rng(seed).normal(size=(batch, 17, 32)).astype(np.float32)
            expected = values @ weight
            expected = ((expected - expected.mean(axis=-1, keepdims=True)) /
                        np.sqrt(expected.var(axis=-1, keepdims=True) + 1e-5))
            with fp32_inference_policy():
                actual = stub.run_numpy({"x": values})["y"]
            np.testing.assert_allclose(actual, expected, atol=1e-4, rtol=1e-4)

        check(models[1], 2, 100)
        for index, batch in enumerate([1, 7, 3, 16] * 4):
            check(models[0], batch, index)
            captures = runtime.graph_capture_count()
            check(models[1], 2, index + 200)
            self.assertEqual(models[1].execution_stats()["last_execution"], "replay")
            self.assertEqual(runtime.graph_capture_count(), captures)


if __name__ == "__main__":
    unittest.main()
