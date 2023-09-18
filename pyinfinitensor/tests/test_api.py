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
        dims = [2, 3, 5, 4]
        np_array = np.random.random(dims).astype(np.float32)
        handler = backend.GraphHandler(backend.cpu_runtime())
        tensor1 = handler.tensor(dims, TensorProto.FLOAT)
        tensor2 = handler.tensor(dims, TensorProto.FLOAT)
        handler.data_malloc()
        tensor1.copyin_float(np_array.flatten().tolist())
        tensor2.copyin_float(np_array.flatten().tolist())
        array1 = np.array(tensor1.copyout_float()).reshape(dims)
        array2 = tensor2.copyout_numpy()
        self.assertTrue(np.array_equal(array2, np_array))
        self.assertTrue(np.array_equal(array1, array2))

        np_array = np.random.random(dims).astype(np.float16)
        np_array[0, 0, 0, 0] = .1
        handler = backend.GraphHandler(backend.cpu_runtime())
        tensor1 = handler.tensor(dims, TensorProto.FLOAT16)
        handler.data_malloc()
        tensor1.copyin_numpy(np_array)
        array1 = tensor1.copyout_numpy()
        # Copy should be the same as original array
        self.assertTrue(np.array_equal(array1, np_array)) 
        # Modify the value so that tensorObj value changes
        np_array[0, 0, 0, 0] = 0. 
        tensor1.copyin_numpy(np_array)
        # The copied-out array should not change
        self.assertFalse(np.array_equal(array1, np_array)) 


if __name__ == "__main__":
    unittest.main()
