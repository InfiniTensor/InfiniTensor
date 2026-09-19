"""A real torch.onnx export, including computed Slice bounds and shape Gather.

Run with -s to retain model provenance, node counts and each numerical result.
"""

import json
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import onnx
import onnxsim

from pyinfinitensor import backend
from pyinfinitensor import onnx as onnx_frontend
from pyinfinitensor.onnx import OnnxStub

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "examples" / "python"))

RTOL = 1e-4
ATOL = 1e-5


class TestRealAttention(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import onnxruntime
            import dynamic_shape_attention
        except ModuleNotFoundError as exc:
            if exc.name in ("torch", "onnxruntime"):
                raise unittest.SkipTest("{} is not installed".format(exc.name)) from exc
            raise

        cls.attention = dynamic_shape_attention
        model = cls.attention.export_attention_model()
        cls.model_bytes = model.SerializeToString()
        options = onnxruntime.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        cls.reference = onnxruntime.InferenceSession(
            cls.model_bytes, options, providers=["CPUExecutionProvider"]
        )
        metadata = cls.attention.export_metadata(model)
        metadata["versions"].update(
            onnxruntime=onnxruntime.__version__, onnxsim=onnxsim.__version__
        )
        metadata["backend"] = backend.__file__
        print("ATTENTION_MODEL " + json.dumps(metadata, sort_keys=True), flush=True)

    def _stub(self, simplify):
        # The frontend renames nodes during import, so every import owns a copy.
        model = onnx.load_model_from_string(self.model_bytes)
        if simplify:
            return OnnxStub(model, backend.cpu_runtime())
        with patch.object(onnx_frontend, "simplify", side_effect=lambda m: (m, False)):
            return OnnxStub(model, backend.cpu_runtime())

    def _serve(self, stub, data, context):
        print("ATTENTION_CASE " + context, flush=True)
        stub.set_input([list(data.shape)])
        stub.inputs["x"].copyin_numpy(data)
        stub.run()
        tensor = stub.outputs["y"]
        self.assertEqual(tuple(tensor.shape()), data.shape, context)
        # Own the result before another shape can reuse the graph's storage.
        return np.asarray(tensor.copyout_float(), dtype=np.float32).reshape(data.shape)

    def _compare_ort(self, got, data, mode, phase):
        want = self.reference.run(["y"], {"x": data})[0]
        context = "real_attention mode={} phase={} shape={}".format(
            mode, phase, data.shape
        )
        self.assertEqual(got.shape, want.shape, context)
        worst = float(np.max(np.abs(got - want)))
        np.testing.assert_allclose(
            got, want, rtol=RTOL, atol=ATOL, equal_nan=False,
            err_msg="{} max_abs_error={}".format(context, worst),
        )
        print("ATTENTION_RESULT " + json.dumps({
            "mode": mode, "phase": phase, "shape": data.shape,
            "max_abs_ort_error": worst,
        }, sort_keys=True), flush=True)

    def _dynamic(self, simplify):
        mode = "default" if simplify else "no_simplify"
        stub = self._stub(simplify)
        first = None
        for step, (batch, seq) in enumerate(self.attention.SHAPES):
            data = self.attention.attention_input(batch, seq)
            context = "real_attention mode={} phase=dynamic step={} shape={}".format(
                mode, step, data.shape
            )
            got = self._serve(stub, data, context)
            self._compare_ort(got, data, mode, "dynamic")
            if first is None:
                first = got
            elif step == len(self.attention.SHAPES) - 1:
                np.testing.assert_array_equal(got, first, err_msg=context)

    def test_dynamic_shapes_by_default(self):
        self._dynamic(simplify=True)

    def test_dynamic_shapes_without_simplification(self):
        self._dynamic(simplify=False)

    def _fold(self, simplify):
        mode = "default" if simplify else "no_simplify"
        stub = self._stub(simplify)
        inputs = [
            self.attention.attention_input(batch, self.attention.PINNED_SEQ)
            for batch in self.attention.PINNED_BATCHES
        ]
        before = []
        for data in inputs:
            context = "real_attention mode={} phase=before_fold shape={}".format(
                mode, data.shape
            )
            got = self._serve(stub, data, context)
            self._compare_ort(got, data, mode, "before_fold")
            before.append(got)

        stub.pin_dims("x", [1])
        nodes_before = len(stub.handler.operators())
        shape_nodes_before = stub.shape_subgraph_size()
        dropped = stub.fold_shape_subgraph()
        nodes_after = len(stub.handler.operators())
        shape_nodes_after = stub.shape_subgraph_size()
        context = "real_attention mode={} pin=seq:{}".format(
            mode, self.attention.PINNED_SEQ
        )
        self.assertGreater(dropped, 0, context + " did not exercise a fold")
        self.assertEqual(nodes_after, nodes_before - dropped, context)
        self.assertEqual(shape_nodes_after, shape_nodes_before - dropped, context)

        dropped_again = stub.fold_shape_subgraph()
        self.assertEqual(dropped_again, 0, context + " repeated fold")
        self.assertEqual(len(stub.handler.operators()), nodes_after, context)
        self.assertEqual(stub.shape_subgraph_size(), shape_nodes_after, context)

        for data, expected in zip(inputs, before):
            case = context + " phase=after_fold shape={}".format(data.shape)
            got = self._serve(stub, data, case)
            np.testing.assert_array_equal(got, expected, err_msg=case)
            self._compare_ort(got, data, mode, "after_fold")

        with self.assertRaisesRegex(RuntimeError, 'input "x"'):
            stub.set_input([[2, self.attention.PINNED_SEQ + 1, self.attention.DIM]])
        self.assertEqual(stub.inputs["x"].shape(), list(inputs[-1].shape), context)
        got = self._serve(stub, inputs[1], context + " phase=after_rejection")
        np.testing.assert_array_equal(got, before[1], err_msg=context)
        self._compare_ort(got, inputs[1], mode, "after_rejection")
        print("ATTENTION_FOLD " + json.dumps({
            "mode": mode, "pinned_seq": self.attention.PINNED_SEQ,
            "batches": self.attention.PINNED_BATCHES,
            "nodes_before": nodes_before, "nodes_after": nodes_after,
            "shape_nodes_before": shape_nodes_before,
            "shape_nodes_after": shape_nodes_after,
            "dropped": dropped, "dropped_again": dropped_again,
            "max_abs_fold_error": 0.0, "pin_rejection_recovered": True,
        }, sort_keys=True), flush=True)

    def test_fold_with_pinned_sequence_by_default(self):
        self._fold(simplify=True)

    def test_fold_with_pinned_sequence_without_simplification(self):
        self._fold(simplify=False)


if __name__ == "__main__":
    unittest.main()
