import os
import unittest

import numpy as np
from onnx import TensorProto

from pyinfinitensor import backend


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


if __name__ == "__main__":
    unittest.main()
