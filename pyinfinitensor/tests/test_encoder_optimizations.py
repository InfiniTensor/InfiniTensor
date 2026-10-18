import os
import unittest
from unittest.mock import patch

import numpy as np
from onnx import TensorProto, helper, numpy_helper

from pyinfinitensor import backend
from pyinfinitensor.embedding import fp32_inference_policy
from pyinfinitensor.onnx import OnnxStub, _linear_bias_patterns


def model(nodes, initializers, outputs=('y',)):
    x = helper.make_tensor_value_info('x', TensorProto.FLOAT, ['B', 4])
    values = [helper.make_tensor_value_info(name, TensorProto.FLOAT, ['B', 4]) for name in outputs]
    return helper.make_model(helper.make_graph(nodes, 'encoder_optimizations', [x], values,
        initializer=[numpy_helper.from_array(np.asarray(value), name) for name, value in initializers.items()]),
        opset_imports=[helper.make_opsetid('', 17)])


class TestLinearPattern(unittest.TestCase):
    def base(self, extra=False, exposed=False, bad_bias=False):
        nodes = [helper.make_node('MatMul', ['x', 'w'], ['product']),
                 helper.make_node('Add', ['bias', 'product'], ['y'])]
        if extra:
            nodes.append(helper.make_node('Identity', ['product'], ['side']))
        return model(nodes, {'w': np.eye(4, dtype=np.float32),
            'bias': np.ones((1, 4) if bad_bias else (4,), dtype=np.float32)},
            ('y', 'product') if exposed else ('y',))

    def test_match_and_reject_unsafe_fusions(self):
        self.assertEqual(_linear_bias_patterns(self.base()), {1: (0, 'bias')})
        self.assertEqual(_linear_bias_patterns(self.base(extra=True)), {})
        self.assertEqual(_linear_bias_patterns(self.base(exposed=True)), {})
        self.assertEqual(_linear_bias_patterns(self.base(bad_bias=True)), {})


class TestEncoderOptimizations(unittest.TestCase):
    def make_stub(self, graph, views, fusion, mode='normal'):
        with patch('pyinfinitensor.onnx.simplify', side_effect=lambda candidate: (candidate, False)):
            return OnnxStub(graph, backend.runtime(backend.default_infini_device()),
                memory_views=views, fuse_linear_bias=fusion, execution_mode=mode)

    def test_views_preserve_fanout_lifetimes_and_output_ownership(self):
        graph = model([
            helper.make_node('MatMul', ['x', 'w'], ['p']),
            helper.make_node('Reshape', ['p', 'shape3'], ['r']),
            helper.make_node('Identity', ['r'], ['r2']),
            helper.make_node('Add', ['r2', 'bias3'], ['a']),
            helper.make_node('Add', ['p', 'bias4'], ['late']),
            helper.make_node('Reshape', ['a', 'shape2'], ['flat']),
            helper.make_node('Add', ['late', 'flat'], ['y']),
        ], {'w': np.eye(4, dtype=np.float32)*2,
            'shape3': np.asarray([0, 2, 2], np.int64), 'shape2': np.asarray([0, 4], np.int64),
            'bias3': np.asarray([[1, 2], [3, 4]], np.float32),
            'bias4': np.arange(4, dtype=np.float32)})
        stub = self.make_stub(graph, True, False, 'graph')
        previous = None
        with fp32_inference_policy():
            for batch in [1, 1, 7, 7, 3, 3, 16, 16, 1, 1]:
                x = np.arange(batch*4, dtype=np.float32).reshape(batch, 4) + batch
                out = stub.run_numpy({'x': x})['y']
                np.testing.assert_allclose(out, 4*x + np.asarray([1, 3, 5, 7], np.float32), atol=1e-5)
                self.assertEqual(stub.tensors['p'].data_ptr(), stub.tensors['r'].data_ptr())
                self.assertEqual(stub.tensors['p'].data_ptr(), stub.tensors['r2'].data_ptr())
                self.assertEqual(stub.tensors['a'].data_ptr(), stub.tensors['flat'].data_ptr())
                self.assertEqual(stub.memory_stats()['view_ops'], 3)
                if previous is not None:
                    np.testing.assert_array_equal(*previous)
                previous = out, out.copy()
        self.assertGreater(stub.execution_stats()['replays'], 0)

    def test_fused_projection_matches_reference_on_dynamic_rows(self):
        rng = np.random.default_rng(318)
        weight = rng.normal(size=(4, 4)).astype(np.float32)
        bias = rng.normal(size=4).astype(np.float32)
        graph = model([helper.make_node('MatMul', ['x', 'w'], ['p']),
                       helper.make_node('Add', ['p', 'bias'], ['y'])], {'w': weight, 'bias': bias})
        stub = self.make_stub(graph, False, True, 'graph')
        with fp32_inference_policy():
            for batch in [1, 1, 8, 8, 3, 3, 32, 32]:
                x = rng.normal(size=(batch, 4)).astype(np.float32)
                np.testing.assert_allclose(stub.run_numpy({'x': x})['y'], x@weight+bias, atol=1e-5, rtol=1e-5)
        self.assertEqual(stub.check_capabilities()['linear_bias_fusions'], 1)

    def test_public_output_keeps_separate_storage(self):
        graph = model([helper.make_node('MatMul', ['x', 'w'], ['p']),
                       helper.make_node('Identity', ['p'], ['y'])], {'w': np.eye(4, dtype=np.float32)})
        stub = self.make_stub(graph, True, False)
        stub.run_numpy({'x': np.ones((2, 4), np.float32)})
        self.assertNotEqual(stub.tensors['p'].data_ptr(), stub.outputs['y'].data_ptr())
        self.assertEqual(stub.memory_stats()['view_ops'], 0)

    def test_tanh_bridge_and_capability_report(self):
        graph = model([helper.make_node('Tanh', ['x'], ['y'])], {})
        stub = self.make_stub(graph, False, False)
        x = np.asarray([[-3., -.5, 0., 2.]], np.float32)
        np.testing.assert_allclose(stub.run_numpy({'x': x})['y'], np.tanh(x), atol=1e-6)
        self.assertIn('Tanh', stub.check_capabilities()['required_ops'])


if __name__ == '__main__':
    unittest.main()
