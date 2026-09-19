import os
import unittest
from unittest.mock import patch

import numpy as np
from onnx import TensorProto, checker, helper, numpy_helper

from pyinfinitensor import backend
from pyinfinitensor import onnx as onnx_frontend


@unittest.skipUnless(hasattr(backend, "InfiniRuntime"), "Infini backend not built")
class TestCopyGraphCapture(unittest.TestCase):
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

    def test_onnx_copy_operators_capture_and_replay(self):
        cases = (
            ("Reshape", [3, 2], "shape", [3, 2], {}),
            ("Flatten", [2, 3], None, None, {"axis": 1}),
            ("Identity", [2, 1, 3], None, None, {}),
            ("Squeeze", [2, 3], "axes", [1], {}),
            ("Unsqueeze", [1, 2, 1, 3], "axes", [0], {}),
        )
        dtypes = (
            (np.float32, TensorProto.FLOAT),
            (np.float16, TensorProto.FLOAT16),
            (np.int32, TensorProto.INT32),
        )
        for op, shape, parameter, parameter_value, attributes in cases:
            for numpy_dtype, onnx_dtype in dtypes:
                for naive in (False, True):
                    with self.subTest(op=op, dtype=numpy_dtype, naive=naive):
                        initializers = []
                        inputs = ["x"]
                        if parameter is not None:
                            inputs.append(parameter)
                            initializers.append(
                                numpy_helper.from_array(
                                    np.asarray(parameter_value, dtype=np.int64),
                                    parameter,
                                )
                            )
                        graph = helper.make_graph(
                            [helper.make_node(op, inputs, ["y"], **attributes)],
                            "copy_capture",
                            [helper.make_tensor_value_info("x", onnx_dtype, [2, 1, 3])],
                            [helper.make_tensor_value_info("y", onnx_dtype, shape)],
                            initializer=initializers,
                        )
                        model = helper.make_model(
                            graph, opset_imports=[helper.make_opsetid("", 18)]
                        )
                        checker.check_model(model)
                        runtime = backend.runtime(backend.default_infini_device())
                        # Preserve the actual copy node instead of allowing the
                        # simplifier to eliminate the operation under test.
                        with patch.object(
                            onnx_frontend, "simplify", side_effect=lambda m: (m, False)
                        ):
                            stub = onnx_frontend.OnnxStub(
                                model, runtime, use_naive_allocator=naive
                            )
                        self.assertEqual(len(stub.handler.operators()), 1)
                        for iteration in range(4):
                            values = (
                                np.arange(6, dtype=numpy_dtype) + iteration * 10 - 7
                            ).reshape(2, 1, 3)
                            stub.inputs["x"].copyin_numpy(values)
                            if iteration == 0:
                                stub.run()
                            else:
                                stub.run_with_graph()
                                self.assertEqual(runtime.graph_capture_count(), 1)
                                self.assertEqual(runtime.graph_cache_size(), 1)
                            actual = stub.outputs["y"].copyout_numpy()
                            self.assertEqual(actual.shape, tuple(shape))
                            self.assertEqual(actual.dtype, numpy_dtype)
                            np.testing.assert_array_equal(actual, values.reshape(shape))


if __name__ == "__main__":
    unittest.main()
