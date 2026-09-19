"""Standard torchvision ResNet18 on one instance with changing batch/H/W."""

import json
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import onnx

from pyinfinitensor import onnx as frontend
from pyinfinitensor.onnx import OnnxStub, backend

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "examples" / "python"))


class TestRealResNet(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import onnxruntime
            import dynamic_shape_resnet
        except ModuleNotFoundError as exc:
            if exc.name in ("torch", "torchvision", "onnxruntime"):
                raise unittest.SkipTest(f"{exc.name} is not installed") from exc
            raise

        cls.source = dynamic_shape_resnet
        model = cls.source.export_resnet_model()
        cls.raw = model.SerializeToString()
        cls.conv_output = next(
            n.output[0] for n in model.graph.node if n.op_type == "Conv"
        )
        cls.gap_output = next(
            n.output[0] for n in model.graph.node if n.op_type == "GlobalAveragePool"
        )
        options = onnxruntime.SessionOptions()
        options.intra_op_num_threads = options.inter_op_num_threads = 1
        cls.reference = onnxruntime.InferenceSession(
            cls.raw, options, providers=["CPUExecutionProvider"]
        )
        metadata = cls.source.export_metadata(model)
        metadata["versions"]["onnxruntime"] = onnxruntime.__version__
        metadata["backend"] = backend.__file__
        print("RESNET_MODEL " + json.dumps(metadata, sort_keys=True), flush=True)

    def _dynamic(self, simplify):
        model = onnx.load_model_from_string(self.raw)
        kwargs = {"input_shapes": {"images": list(self.source.SHAPES[0])}}
        mode = "default" if simplify else "no_simplify"
        if simplify:
            stub = OnnxStub(model, backend.cpu_runtime(), **kwargs)
        else:
            with patch.object(frontend, "simplify", side_effect=lambda m: (m, False)):
                stub = OnnxStub(model, backend.cpu_runtime(), **kwargs)
        first = None
        for step, shape in enumerate(self.source.SHAPES):
            context = f"ResNet18 mode={mode} step={step} shape={shape}"
            print("RESNET_CASE " + context, flush=True)
            data = self.source.resnet_input(shape)
            stub.set_input([list(shape)])
            self.assertEqual(
                stub.tensors[self.conv_output].shape(),
                [shape[0], 64, (shape[2] + 1) // 2, (shape[3] + 1) // 2],
                context,
            )
            self.assertEqual(
                stub.tensors[self.gap_output].shape(), [shape[0], 512, 1, 1], context
            )
            stub.inputs["images"].copyin_numpy(data)
            stub.run()
            got = stub.outputs["logits"].copyout_numpy().copy()
            want = self.reference.run(["logits"], {"images": data})[0]
            self.assertEqual(got.shape, (shape[0], 1000), context)
            self.assertEqual(got.shape, want.shape, context)
            np.testing.assert_allclose(
                got, want, rtol=1e-4, atol=1e-5, equal_nan=False, err_msg=context
            )
            if first is None:
                first = got
            elif step == len(self.source.SHAPES) - 1:
                np.testing.assert_array_equal(got, first, err_msg=context)
            print(
                "RESNET_RESULT "
                + json.dumps(
                    {
                        "mode": mode,
                        "shape": shape,
                        "output_shape": got.shape,
                        "max_abs_ort_error": float(np.max(np.abs(got - want))),
                        "nodes": len(stub.handler.operators()),
                        "shape_nodes": stub.shape_subgraph_size(),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

    def test_dynamic_shapes_by_default(self):
        self._dynamic(True)

    def test_dynamic_shapes_without_simplification(self):
        self._dynamic(False)


if __name__ == "__main__":
    unittest.main()
