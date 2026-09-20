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
                if iteration < 2:
                    handler.run()
                else:
                    handler.run_with_graph()
                    self.assertEqual(runtime.graph_capture_count(), 1)
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

    def test_batch_norm_max_pool_dynamic_shape_and_replay(self):
        # The standalone C++ test cannot initialize the Python-based NPU
        # provider. Cover its dynamic scratch sizes and both allocators here.
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
                captures = 0
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
                            handler.run_with_graph()
                            captures += iteration == 0
                            self.assertEqual(runtime.graph_capture_count(), captures)
                            tolerance = 3e-3 if dtype == np.float16 else 1e-5
                            np.testing.assert_allclose(y.copyout_numpy(), expected,
                                                       rtol=tolerance, atol=tolerance)
                finally:
                    runtime.clear_graph_cache()

    def check_transform_capture_pool(self, transform):
        runtime = backend.runtime("ascend", 0)
        handler = backend.GraphHandler(runtime)
        shape = [256, 1024] if transform == "transpose" else [1, 1024]
        x = handler.tensor(shape, 1)
        x.set_input()
        if transform == "transpose":
            y = handler.transpose(x, None, [1, 0])
        else:
            y = handler.expand(x, None, [256, 1024])
        y.set_output()
        handler.data_malloc()
        values = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
        x.copyin_numpy(values)
        handler.run()
        torch.npu.empty_cache()
        reserved_before = torch.npu.memory_reserved()

        pressure = []
        try:
            for iteration in range(6):
                current = values + iteration * 17
                x.copyin_numpy(current)
                handler.run_with_graph()
                self.assertEqual(runtime.graph_capture_count(), 1)
                expected = (current.T if transform == "transpose"
                            else np.broadcast_to(current, [256, 1024]))
                np.testing.assert_array_equal(y.copyout_numpy(), expected)
                # A replay must not overwrite an unrelated allocation that
                # reused a formerly cached address after empty_cache().
                for allocation in pressure:
                    np.testing.assert_array_equal(allocation.cpu().numpy(), -999.0)
                if pressure:
                    del allocation
                # Both generated out variants allocate an intermediate tensor.
                # Freeing ordinary cached blocks must leave its graph pool live.
                torch.npu.empty_cache()
                self.assertGreater(torch.npu.memory_reserved(), reserved_before)
                if iteration == 0:
                    pressure = [torch.full((256, 1024), -999.0, device="npu")
                                for _ in range(64)]
                    # Finish sentinel initialization on the PyTorch stream
                    # before checking writes from the separate runtime stream.
                    torch.npu.current_stream().synchronize()
        finally:
            runtime.clear_graph_cache()
            pressure.clear()
        torch.npu.empty_cache()
        self.assertEqual(torch.npu.memory_reserved(), reserved_before)

    def test_transpose_replay_after_empty_cache(self):
        self.check_transform_capture_pool("transpose")

    def test_expand_replay_after_empty_cache(self):
        self.check_transform_capture_pool("expand")

    def test_capture_pool_released_on_eviction(self):
        runtime = backend.runtime("ascend", 0, 1)
        graphs = []
        values = np.arange(256 * 1024, dtype=np.float32).reshape(256, 1024)
        for _ in range(2):
            handler = backend.GraphHandler(runtime)
            x = handler.tensor([256, 1024], 1)
            x.set_input()
            y = handler.transpose(x, None, [1, 0])
            y.set_output()
            handler.data_malloc()
            x.copyin_numpy(values)
            handler.run()
            graphs.append((handler, x, y))
        torch.npu.empty_cache()
        reserved_before = torch.npu.memory_reserved()

        try:
            for iteration in range(6):
                handler, x, y = graphs[iteration % 2]
                current = values + iteration * 23
                x.copyin_numpy(current)
                handler.run_with_graph()
                np.testing.assert_array_equal(y.copyout_numpy(), current.T)
                self.assertEqual(runtime.graph_cache_size(), 1)
                self.assertEqual(runtime.graph_capture_count(), iteration + 1)
                torch.npu.empty_cache()
                if iteration == 0:
                    reserved_with_one_graph = torch.npu.memory_reserved()
                    self.assertGreater(reserved_with_one_graph, reserved_before)
                else:
                    self.assertEqual(torch.npu.memory_reserved(), reserved_with_one_graph)
        finally:
            runtime.clear_graph_cache()
        torch.npu.empty_cache()
        self.assertEqual(torch.npu.memory_reserved(), reserved_before)

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
