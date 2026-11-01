"""Run with an Ascend ATen build's backend module on PYTHONPATH.

Requires matching torch/torch_npu packages with external-stream support
(validated with torch 2.7.1 and torch_npu 2.7.1.post10), and
TASK_QUEUE_ENABLE=0 set before starting Python.
"""

import os
import pathlib
import subprocess
import sys
import unittest

import numpy as np
import torch
import torch_npu  # noqa: F401

import backend


class AscendAtenTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.npu.config.allow_internal_format = False
        torch.npu.init()

    def check_relu(self, with_copy):
        runtime = backend.runtime("ascend", 0)
        handler = backend.GraphHandler(runtime)
        x = handler.tensor([32, 128], 1)
        x.set_input()
        y = handler.relu(x, None)
        if with_copy:
            # InfiniRT D2D and ATen kernels must share the runtime's stream.
            y = handler.identity(y, None)
            y = handler.relu(y, None)
        y.set_output()
        handler.data_malloc()

        other_stream = torch.npu.Stream()
        with torch.npu.stream(other_stream):
            for iteration in range(22):
                # Alternate signs so replaying a stale result cannot pass.
                values = np.arange(4096, dtype=np.float32).reshape(32, 128)
                values = (values - 2048 + iteration * 17) * (-1) ** iteration
                x.copyin_numpy(values)
                handler.run()
                if iteration == 2:
                    with self.assertRaisesRegex(RuntimeError, "not capture-safe"):
                        handler.run_with_graph()
                    self.assertEqual(runtime.graph_capture_count(), 0)
                    handler.run()
                self.assertEqual(
                    torch.npu.current_stream().npu_stream, other_stream.npu_stream
                )
                # No torch.npu.synchronize(): InfiniTensor must wait for its
                # own stream, including every ATen operation submitted to it.
                np.testing.assert_array_equal(y.copyout_numpy(), np.maximum(values, 0))
        runtime.clear_graph_cache()

    def test_external_allocation_relu(self):
        self.check_relu(with_copy=False)

    def test_relu_copy_relu_stream_order(self):
        self.check_relu(with_copy=True)

    def test_batch_norm_max_pool_dynamic_shape_and_capture_rejection(self):
        # ATen providers currently reject capture through the public runtime.
        # Ordinary execution must recover and retain correct dynamic outputs.
        cases = ((np.float32, 1, False), (np.float32, 1, True),
                 (np.float16, 10, False))
        for dtype, proto, naive in cases:
            with self.subTest(dtype=dtype, naive=naive):
                runtime = backend.runtime("ascend", 0, 1)
                handler = backend.GraphHandler(runtime)
                x = handler.tensor([1, 2, 4, 4], proto)
                x.set_input()
                parameters = [handler.tensor([2], 1) for _ in range(4)]
                for tensor in parameters:
                    tensor.set_input()
                bn = handler.batchNormalization(
                    x, None, *parameters, 0.9, 1e-5, False)
                y = handler.maxPool(bn, None, 2, 2, 1, 1, 0, 0, 2, 2, 0)
                y.set_output()
                try:
                    for channels in (2, 3, 2):
                        handler.change_shape([1, channels, 4, 4], x.fuid())
                        for tensor in parameters:
                            handler.change_shape([channels], tensor.fuid())
                        handler.shape_infer()
                        handler.data_malloc(naive)
                        for iteration in range(3):
                            values = (np.arange(channels * 16, dtype=np.float32)
                                      - 10 + iteration).astype(dtype).reshape(1, channels, 4, 4)
                            x.copyin_numpy(values)
                            for tensor, value in zip(parameters, (iteration, 4, 2, -1)):
                                tensor.copyin_numpy(np.full(channels, value, np.float32))
                            expected = ((values.astype(np.float32) - iteration)
                                        / np.sqrt(4 + 1e-5) * 2 - 1).astype(dtype)
                            expected = expected.reshape(1, channels, 2, 2, 2, 2).max(axis=(3, 5))
                            handler.run()
                            if iteration == 0:
                                with self.assertRaisesRegex(RuntimeError, "not capture-safe"):
                                    handler.run_with_graph()
                                self.assertEqual(runtime.graph_capture_count(), 0)
                                handler.run()
                            tolerance = 3e-3 if dtype == np.float16 else 1e-5
                            np.testing.assert_allclose(y.copyout_numpy(), expected,
                                                       rtol=tolerance, atol=tolerance)
                finally:
                    runtime.clear_graph_cache()

    def test_rejects_async_host_queue(self):
        env = dict(os.environ, TASK_QUEUE_ENABLE="2")
        result = subprocess.run(
            [sys.executable, str(pathlib.Path(__file__).resolve()),
             "AscendAtenTest.test_external_allocation_relu"],
            env=env, capture_output=True, text=True, timeout=90,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("external streams require TASK_QUEUE_ENABLE=0", result.stderr)


if __name__ == "__main__":
    unittest.main()
