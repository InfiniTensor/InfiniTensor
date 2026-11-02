"""Accelerator ATen regression; missing SDPA or device capabilities must fail."""

import unittest

import numpy as np
from pyinfinitensor import backend
import torch


class TestInfiniOpsStream(unittest.TestCase):
    def test_sdpa_identity_uses_runtime_stream(self):
        name = backend.default_infini_device()
        device = {"nvidia": "cuda", "cambricon": "mlu", "ascend": "npu"}[name]
        accelerator = getattr(torch, device)
        runtime = backend.InfiniRuntime(name, 0)
        handler = backend.GraphHandler(runtime)
        shape = [2, 16, 32]
        inputs = [handler.tensor(shape, 1) for _ in range(3)]
        intermediate = handler.tensor(shape, 1)
        for tensor in inputs:
            tensor.set_input()
        intermediate.set_output()
        handler.infini_op(
            "ScaledDotProductAttention",
            dict(zip(("query", "key", "value"), ([t] for t in inputs))),
            {"out": [intermediate]},
        )
        output = handler.identity(intermediate, None)
        output.set_output()
        handler.data_malloc()
        rng = np.random.default_rng(327)

        def load():
            arrays = [rng.uniform(-0.5, 0.5, shape).astype(np.float32) for _ in inputs]
            for tensor, array in zip(inputs, arrays):
                tensor.copyin_numpy(array)
            intermediate.copyin_numpy(np.full(shape, -123, np.float32))
            scores = arrays[0] @ arrays[1].swapaxes(-1, -2) / np.sqrt(shape[-1])
            weights = np.exp(scores - scores.max(axis=-1, keepdims=True))
            return weights / weights.sum(axis=-1, keepdims=True) @ arrays[2]

        expected = load()
        handler.run()  # Initialize the provider before introducing delayed work.
        np.testing.assert_allclose(output.copyout_numpy(), expected, atol=1e-4, rtol=1e-4)
        other = accelerator.Stream()
        matrix = torch.ones((1024, 1024), device=device)
        scratch = torch.empty_like(matrix)
        other.wait_stream(accelerator.current_stream())
        try:
            for _ in range(6):
                expected = load()
                with accelerator.stream(other):
                    if device == "cuda":
                        accelerator._sleep(100_000_000)
                    else:
                        for _ in range(64):
                            torch.mm(matrix, matrix, out=scratch)
                    handler.run()
                    self.assertEqual(accelerator.current_stream(), other)
                    # No device or Torch stream synchronization before checking
                    # the RT copy that consumes the ATen SDPA output.
                    np.testing.assert_allclose(
                        output.copyout_numpy(), expected, atol=1e-4, rtol=1e-4
                    )
        finally:
            # Cleanup only, after output assertions; never used for correctness.
            other.synchronize()


if __name__ == "__main__":
    unittest.main()
