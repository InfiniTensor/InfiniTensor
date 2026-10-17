import unittest
from unittest.mock import patch

import numpy as np
from onnx import helper

from pyinfinitensor import backend
from test_onnxstub import import_model, make_model, value_info


class FakeRuntime:
    def __init__(self):
        self.captures = 0
        self.ready = False

    def graph_capture_count(self):
        return self.captures


class TestExecutionPolicy(unittest.TestCase):
    def make_stub(self):
        stub = import_model(make_model(
            [helper.make_node("Identity", ["x"], ["y"])],
            [value_info("x", ["B", 2])], [value_info("y", ["B", 2])]))
        real = stub.handler
        runtime = FakeRuntime()

        class Handler:
            fail = False

            def __getattr__(self, name):
                return getattr(real, name)

            def clear_graph_cache(self):
                runtime.ready = False
                real.clear_graph_cache()

            def run_with_graph(self):
                if self.fail:
                    raise RuntimeError("injected capture failure")
                if not runtime.ready:
                    runtime.captures += 1
                    runtime.ready = True
                real.run()

        stub.handler, stub._runtime = Handler(), runtime
        return stub

    def test_auto_replays_new_contents_and_resets_after_shape_change(self):
        stub = self.make_stub()
        with patch.object(backend, "InfiniRuntime", FakeRuntime, create=True):
            stub.set_execution_mode("auto", 3)
        prior = None
        for shape, expected in [((1, 2), "normal"), ((1, 2), "normal"),
                                ((1, 2), "capture"), ((1, 2), "replay"),
                                ((3, 2), "normal"), ((3, 2), "normal"),
                                ((3, 2), "capture"), ((1, 2), "normal")]:
            values = np.full(shape, stub.execution_stats()["normal_runs"] +
                             stub.execution_stats()["replays"] + 1, np.float32)
            out = stub.run_numpy({"x": values})["y"]
            np.testing.assert_array_equal(out, values)
            self.assertEqual(stub.execution_stats()["last_execution"], expected)
            if prior is not None:
                np.testing.assert_array_equal(*prior)
            prior = out, out.copy()
        self.assertEqual(stub.execution_stats()["captures"], 2)

    def test_capture_failure_propagates_and_normal_execution_can_recover(self):
        stub = self.make_stub()
        with patch.object(backend, "InfiniRuntime", FakeRuntime, create=True):
            stub.set_execution_mode("graph", 1)
        stub.handler.fail = True
        values = np.ones((1, 2), np.float32)
        with self.assertRaisesRegex(RuntimeError, "injected capture failure"):
            stub.run_numpy({"x": values})
        self.assertEqual(stub.execution_stats()["normal_runs"], 0)
        self.assertEqual(stub.execution_stats()["graph_failures"], 1)
        stub.set_execution_mode("normal")
        np.testing.assert_array_equal(stub.run_numpy({"x": values})["y"], values)

    def test_mode_validation_does_not_change_policy(self):
        stub = self.make_stub()
        for mode, after in [("cuda", 2), ("normal", 0), ("normal", 1.5),
                            ("normal", True)]:
            with self.assertRaises(ValueError):
                stub.set_execution_mode(mode, after)
        self.assertEqual(stub.execution_mode, "normal")

    def test_unprepared_graph_execution_is_rejected(self):
        stub = self.make_stub()
        stub._prepared_is_valid = False
        with self.assertRaisesRegex(RuntimeError, "not been prepared"):
            stub.run_with_graph()


if __name__ == "__main__":
    unittest.main()
