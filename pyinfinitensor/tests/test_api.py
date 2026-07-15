import os, unittest
from onnx import TensorProto
from pyinfinitensor import backend
import numpy as np


class TestPythonAPI(unittest.TestCase):
    def test_copyin_numpy(self):
        dims = [2, 3, 5, 4]
        np_array = np.random.random(dims).astype(np.float32)
        handler = backend.GraphHandler(backend.cpu_runtime())
        tensor1 = handler.tensor(dims, TensorProto.FLOAT)
        tensor2 = handler.tensor(dims, TensorProto.FLOAT)
        handler.data_malloc()
        tensor1.copyin_numpy(np_array)
        tensor2.copyin_float(np_array.flatten().tolist())
        array1 = tensor1.copyout_float()
        array2 = tensor2.copyout_float()
        self.assertEqual(array1, array2)
        self.assertTrue(np.array_equal(np.array(array1).reshape(dims), np_array))

        np_array = np.random.random(dims).astype(np.int64)
        handler = backend.GraphHandler(backend.cpu_runtime())
        tensor1 = handler.tensor(dims, TensorProto.INT64)
        tensor2 = handler.tensor(dims, TensorProto.INT64)
        handler.data_malloc()
        tensor1.copyin_numpy(np_array)
        tensor2.copyin_int64(np_array.flatten().tolist())
        array1 = tensor1.copyout_int64()
        array2 = tensor2.copyout_int64()
        self.assertEqual(array1, array2)
        self.assertTrue(np.array_equal(np.array(array1).reshape(dims), np_array))

    def test_copyout_numpy(self):
        dims = [2, 1, 3, 4]
        size = np.prod(dims)
        cases = [
            (np.arange(size, dtype=np.float32).reshape(dims), TensorProto.FLOAT),
            (np.arange(size, dtype=np.float16).reshape(dims), TensorProto.FLOAT16),
            (np.arange(size, dtype=np.int32).reshape(dims), TensorProto.INT32),
        ]

        for expected, tensor_dtype in cases:
            with self.subTest(dtype=expected.dtype):
                handler = backend.GraphHandler(backend.cpu_runtime())
                tensor = handler.tensor(dims, tensor_dtype)
                handler.data_malloc()
                tensor.copyin_numpy(expected)

                actual = tensor.copyout_numpy()
                self.assertEqual(actual.shape, expected.shape)
                self.assertEqual(actual.dtype, expected.dtype)
                self.assertEqual(actual.strides, expected.strides)
                self.assertTrue(actual.flags.c_contiguous)
                self.assertTrue(np.array_equal(actual, expected))

                tensor.copyin_numpy(np.zeros_like(expected))
                self.assertTrue(np.array_equal(actual, expected))


if __name__ == "__main__":
    unittest.main()
