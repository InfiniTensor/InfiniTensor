import unittest
from unittest.mock import Mock, patch

import numpy as np
from onnx import TensorProto, checker, helper, numpy_helper

from pyinfinitensor import backend
from pyinfinitensor import onnx as onnx_frontend
from pyinfinitensor.onnx import OnnxStub, _parse_attribute


def make_model(nodes, inputs, outputs, initializers=(), check=True):
    graph = helper.make_graph(
        nodes,
        "onnxstub_test",
        inputs,
        outputs,
        initializer=list(initializers),
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
    if check:
        checker.check_model(model)
    return model


def value_info(name, shape, dtype=TensorProto.FLOAT):
    return helper.make_tensor_value_info(name, dtype, shape)


def initializer(name, values, dtype=None):
    array = np.asarray(values, dtype=dtype)
    return numpy_helper.from_array(array, name=name)


def import_model(model, runtime=None, use_naive_allocator=False, input_shapes=None):
    runtime = backend.cpu_runtime() if runtime is None else runtime
    with patch.object(
        onnx_frontend, "simplify", side_effect=lambda candidate: (candidate, False)
    ):
        return OnnxStub(
            model,
            runtime,
            use_naive_allocator=use_naive_allocator,
            input_shapes=input_shapes,
        )


def node_attribute(node, name):
    attribute = next(attr for attr in node.attribute if attr.name == name)
    return helper.get_attribute_value(attribute)


def _reshape_chain_model(tail):
    """A model reshaping `x` to a target its own shape subgraph works out.

    The batch dimension is read off `x` at run time and joined with `tail`, so
    the target is an edge of the graph rather than a constant known when the
    graph is built. A `tail` of -1 leaves the second dimension for Reshape to
    work out from the number of elements.
    """
    return make_model(
        [
            helper.make_node("Shape", ["x"], ["s"]),
            helper.make_node("Gather", ["s", "first"], ["g"], axis=0),
            helper.make_node("Unsqueeze", ["g", "zero"], ["u"]),
            helper.make_node("Concat", ["u", "tail"], ["target"], axis=0),
            helper.make_node("Reshape", ["x", "target"], ["y"]),
        ],
        [value_info("x", ["batch", 3, 5])],
        [value_info("y", ["batch", 15])],
        [
            initializer("first", 0, np.int64),
            initializer("zero", [0], np.int64),
            initializer("tail", tail, np.int64),
        ],
    )


def _expand_chain_model(tail):
    """A model expanding `x` to a target its own shape subgraph works out.

    An exporter writes the target of an expand over a dynamic input this way:
    the batch dimension is read off `x` at run time and joined with `tail`, so
    the target is only a number once a shape has been given.
    """
    return make_model(
        [
            helper.make_node("Shape", ["x"], ["s"]),
            helper.make_node("Gather", ["s", "first"], ["g"], axis=0),
            helper.make_node("Unsqueeze", ["g", "zero"], ["u"]),
            helper.make_node("Concat", ["u", "tail"], ["target"], axis=0),
            helper.make_node("Expand", ["x", "target"], ["y"]),
        ],
        [value_info("x", ["batch", 1, 4])],
        [value_info("y", ["batch", 3, 4])],
        [
            initializer("first", 0, np.int64),
            initializer("zero", [0], np.int64),
            initializer("tail", tail, np.int64),
        ],
    )


def _tile_chain_model():
    """A model tiling `x` by counts its own shape subgraph works out.

    The batch is repeated as many times as the batch is long, which is only a
    number once a shape has been given, while the other axis is repeated once by
    a constant.
    """
    return make_model(
        [
            helper.make_node("Shape", ["x"], ["s"]),
            helper.make_node("Gather", ["s", "first"], ["g"], axis=0),
            helper.make_node("Unsqueeze", ["g", "zero"], ["u"]),
            helper.make_node("Concat", ["u", "one"], ["repeats"], axis=0),
            helper.make_node("Tile", ["x", "repeats"], ["y"]),
        ],
        [value_info("x", ["batch", 3])],
        [value_info("y", ["tiled", 3])],
        [
            initializer("first", 0, np.int64),
            initializer("zero", [0], np.int64),
            initializer("one", [1], np.int64),
        ],
    )


def _dynamic_resize_by_sizes_model(sizes, policy="stretch", axes=None):
    """A model resizing `x` to `sizes` while its spatial dimensions are dynamic.

    The requested size is what the model settled on; the ratio it works out to
    depends on the input, which is not known until a shape arrives.
    """
    kwargs = {"mode": "nearest", "keep_aspect_ratio_policy": policy}
    if axes is not None:
        kwargs["axes"] = axes
    return make_model(
        [helper.make_node("Resize", ["x", "", "", "sizes"], ["y"], **kwargs)],
        [value_info("x", [1, 3, "height", "width"])],
        [value_info("y", None)],
        [initializer("sizes", sizes, np.int64)],
        check=False,
    )


def _dynamic_convtranspose_model():
    return make_model(
        [
            helper.make_node(
                "ConvTranspose",
                ["x", "w"],
                ["y"],
                kernel_shape=[3, 3],
                strides=[2, 2],
                pads=[1, 1, 1, 1],
                output_padding=[1, 1],
            )
        ],
        [value_info("x", [1, 3, "height", "width"])],
        [value_info("y", None)],
        [initializer("w", np.ones((3, 2, 3, 3), dtype=np.float32))],
        check=False,
    )


def _dynamic_conv_model():
    return make_model(
        [helper.make_node("Conv", ["x", "w"], ["y"], kernel_shape=[3, 3])],
        [value_info("x", ["batch", 3, "height", "width"])],
        [value_info("y", ["batch", 2, "out_height", "out_width"])],
        [initializer("w", np.ones((2, 3, 3, 3), dtype=np.float32))],
    )


def _only_reshape(stub):
    reshapes = [
        op
        for op in stub.handler.operators()
        if op.op_type().id() == backend.OpTypeId.Reshape
    ]
    assert len(reshapes) == 1, len(reshapes)
    return reshapes[0]


def _settled_chain_model():
    """A shape subgraph whose every element is settled when the graph is built.

    `x` moves only in its first dimension, so reading the second one off it
    gives a number no legal shape can change. Joined with a constant tail, the
    whole chain is known ahead of any run.
    """
    return make_model(
        [
            helper.make_node("Shape", ["x"], ["s"]),
            helper.make_node("Gather", ["s", "middle"], ["g"], axis=0),
            helper.make_node("Unsqueeze", ["g", "zero"], ["u"]),
            helper.make_node("Concat", ["u", "tail"], ["c"], axis=0),
        ],
        [value_info("x", ["batch", 3, 5])],
        [value_info("c", [2], TensorProto.INT64)],
        [
            initializer("middle", 1, np.int64),
            initializer("zero", [0], np.int64),
            initializer("tail", [7], np.int64),
        ],
    )


def _moving_chain_model():
    """The same chain reading the one dimension that does move."""
    return make_model(
        [
            helper.make_node("Shape", ["x"], ["s"]),
            helper.make_node("Gather", ["s", "first"], ["g"], axis=0),
            helper.make_node("Unsqueeze", ["g", "zero"], ["u"]),
            helper.make_node("Concat", ["u", "tail"], ["c"], axis=0),
        ],
        [value_info("x", ["batch", 3, 5])],
        [value_info("c", [2], TensorProto.INT64)],
        [
            initializer("first", 0, np.int64),
            initializer("zero", [0], np.int64),
            initializer("tail", [7], np.int64),
        ],
    )


def _flatten_chain_model(axis_from_constant=False):
    """A model flattening every dimension of `x` but the last.

    The last dimension is fixed, so that branch of the shape subgraph settles
    when the graph is built, while `-1` leaves the moving part for Reshape to
    work out. This is what an exporter emits for a flatten before a linear
    layer. The axis reaches the graph either as an initializer or as a
    `Constant`, which an exporter picks between freely.
    """
    axis = np.asarray(2, dtype=np.int64)
    nodes = [
        helper.make_node("Shape", ["x"], ["s"]),
        helper.make_node("Gather", ["s", "last"], ["g"], axis=0),
        helper.make_node("Unsqueeze", ["g", "zero"], ["u"]),
        helper.make_node("Concat", ["rest", "u"], ["target"], axis=0),
        helper.make_node("Reshape", ["x", "target"], ["y"]),
    ]
    initializers = [
        initializer("zero", [0], np.int64),
        initializer("rest", [-1], np.int64),
    ]
    if axis_from_constant:
        nodes.insert(
            0,
            helper.make_node(
                "Constant",
                [],
                ["last"],
                value=numpy_helper.from_array(axis, name="last"),
            ),
        )
    else:
        initializers.append(initializer("last", axis))
    return make_model(
        nodes,
        [value_info("x", ["batch", 3, 5])],
        [value_info("y", ["flat", 5])],
        initializers,
    )


class TestOnnxStubImport(unittest.TestCase):
    def test_get_perf_time_returns_backend_value(self):
        stub = OnnxStub.__new__(OnnxStub)
        stub.handler = Mock()
        stub.handler.get_perf_time.return_value = 1.25

        self.assertEqual(stub.get_perf_time(), 1.25)

    def test_model_proto_is_not_modified(self):
        x = value_info("x", [2])
        y = value_info("y", [2])
        nodes = [
            helper.make_node("Identity", ["x"], ["middle"]),
            helper.make_node("Relu", ["middle"], ["y"]),
        ]
        model = make_model(nodes, [x], [y])
        before = model.SerializeToString()

        import_model(model)

        self.assertEqual(model.SerializeToString(), before)

    def test_unsorted_dag_creates_each_node_once(self):
        x = value_info("x", [2])
        y = value_info("y", [2])
        nodes = [
            helper.make_node("Relu", ["middle"], ["y"], name="relu"),
            helper.make_node("Identity", ["x"], ["middle"], name="identity"),
        ]
        model = make_model(nodes, [x], [y], check=False)

        stub = import_model(model)

        self.assertEqual(len(stub.handler.operators()), 2)

    def test_unresolved_graph_reports_nodes_and_inputs(self):
        y = value_info("y", [2])
        missing = make_model(
            [helper.make_node("Identity", ["missing"], ["y"], name="consumer")],
            [],
            [y],
            check=False,
        )
        with self.assertRaisesRegex(ValueError, "consumer.*missing"):
            import_model(missing)

        cycle = make_model(
            [
                helper.make_node("Identity", ["b"], ["a"], name="first"),
                helper.make_node("Identity", ["a"], ["b"], name="second"),
            ],
            [],
            [value_info("a", [2])],
            check=False,
        )
        with self.assertRaisesRegex(ValueError, "first.*b.*second.*a"):
            import_model(cycle)

    def test_attribute_parsing_does_not_share_state(self):
        with_alpha = helper.make_node("LeakyRelu", ["x"], ["y"], alpha=0.25)
        without_alpha = helper.make_node("Relu", ["x"], ["y"])

        self.assertEqual(_parse_attribute(with_alpha)["alpha"], 0.25)
        self.assertNotIn("alpha", _parse_attribute(without_alpha))

        defaults = {"axis": 1}
        parsed = _parse_attribute(with_alpha, defaults)
        parsed["axis"] = 2
        self.assertEqual(defaults, {"axis": 1})

    def test_constant_of_shape_is_explicitly_rejected(self):
        shape = value_info("shape", [2], TensorProto.INT64)
        output = value_info("output", [2, 2])
        model = make_model(
            [helper.make_node("ConstantOfShape", ["shape"], ["output"])],
            [shape],
            [output],
        )

        with self.assertRaisesRegex(NotImplementedError, "ConstantOfShape"):
            import_model(model)

    def test_expand_reads_a_computed_target_under_every_shape(self):
        """A target worked out from the input's own shape is read afresh.

        Taken once at construction it would hold the placeholder batch, and an
        input dimension wider than the target's keeps its own size, so every
        later shape would come out with the placeholder's batch instead of its
        own.
        """
        stub = import_model(_expand_chain_model([3, 4]))
        self.assertEqual(stub.outputs["y"].shape(), [1, 3, 4])
        for batch in (2, 5, 7):
            stub.set_input([[batch, 1, 4]])
            self.assertEqual(
                stub.outputs["y"].shape(),
                [batch, 3, 4],
                "batch {}".format(batch),
            )

    def test_expand_with_a_constant_target_follows_the_input(self):
        """A target of one leaves the input dimension alone, so it is followed.

        A target above one settles the dimension instead: broadcasting either
        agrees with the target or stretches a one up to it.
        """
        model = make_model(
            [helper.make_node("Expand", ["x", "target"], ["y"])],
            [value_info("x", ["batch", 1, 4])],
            [value_info("y", ["batch", 3, 4])],
            [initializer("target", [1, 3, 4], np.int64)],
        )
        stub = import_model(model)
        for batch in (2, 5):
            stub.set_input([[batch, 1, 4]])
            self.assertEqual(
                stub.outputs["y"].shape(),
                [batch, 3, 4],
                "batch {}".format(batch),
            )

    def test_expand_holding_its_target_as_an_edge_survives_a_round_trip(self):
        stub = import_model(_expand_chain_model([3, 4]))

        exported = stub.to_onnx("roundtrip")
        checker.check_model(exported)

        self.assertEqual(
            [node.op_type for node in exported.graph.node],
            ["Shape", "Gather", "Unsqueeze", "Concat", "Expand"],
        )
        # An Expand already holding its target as an edge must not have a second
        # one spelled out as a constant beside it.
        expand = next(node for node in exported.graph.node if node.op_type == "Expand")
        self.assertEqual(len(expand.input), 2)

        # The exported model has to keep working out its own dimensions, which
        # is what makes the round trip meaningful rather than merely valid.
        again = import_model(exported)
        again.set_input([[6, 1, 4]])
        self.assertEqual(
            again.outputs[exported.graph.output[0].name].shape(), [6, 3, 4]
        )

    def test_tile_reads_computed_counts_under_every_shape(self):
        """Counts from an edge are read on each inference, not at import."""
        stub = import_model(_tile_chain_model())
        for batch in (1, 2, 5, 3, 1):
            stub.set_input([[batch, 3]])
            self.assertEqual(
                stub.outputs["y"].shape(),
                [batch * batch, 3],
                "batch {}".format(batch),
            )

    def test_tile_repeats_its_input_where_the_counts_say(self):
        """The numbers that come out, checked against numpy's own tiling."""
        model = make_model(
            [helper.make_node("Tile", ["x", "repeats"], ["y"])],
            [value_info("x", [2, 3])],
            [value_info("y", [4, 6])],
            [initializer("repeats", [2, 2], np.int64)],
        )
        stub = import_model(model)
        data = np.arange(6, dtype=np.float32).reshape(2, 3)
        stub.inputs["x"].copyin_float(data.flatten().tolist())
        stub.run()
        actual = np.asarray(stub.outputs["y"].copyout_float()).reshape(4, 6)
        np.testing.assert_array_equal(actual, np.tile(data, (2, 2)))

    def test_tile_with_constant_counts_follows_the_input(self):
        """A count is a multiple of the input, so the output tracks it."""
        model = make_model(
            [helper.make_node("Tile", ["x", "repeats"], ["y"])],
            [value_info("x", ["batch", 3])],
            [value_info("y", ["tiled", 6])],
            [initializer("repeats", [3, 2], np.int64)],
        )
        stub = import_model(model)
        for batch in (2, 5):
            stub.set_input([[batch, 3]])
            self.assertEqual(
                stub.outputs["y"].shape(), [batch * 3, 6], "batch {}".format(batch)
            )

    def test_tile_holding_its_counts_as_an_edge_survives_a_round_trip(self):
        stub = import_model(_tile_chain_model())

        exported = stub.to_onnx("roundtrip")
        checker.check_model(exported)

        self.assertEqual(
            [node.op_type for node in exported.graph.node],
            ["Shape", "Gather", "Unsqueeze", "Concat", "Tile"],
        )
        # A Tile already holding its counts as an edge must not have a second
        # set spelled out as a constant beside it.
        tile = next(node for node in exported.graph.node if node.op_type == "Tile")
        self.assertEqual(len(tile.input), 2)

        again = import_model(exported)
        again.set_input([[4, 3]])
        self.assertEqual(again.outputs[exported.graph.output[0].name].shape(), [16, 3])

    def test_resize_to_sizes_reaches_them_whatever_the_input_was(self):
        """A requested size is reached under every input, not scaled by it.

        The scale is the ratio of the requested size to the input dimension, so
        taking it once at construction -- where a dynamic dimension is still the
        placeholder one -- would resize each later shape by that ratio instead of
        to the size the model asked for.
        """
        stub = import_model(_dynamic_resize_by_sizes_model([1, 3, 8, 8]))
        for shape in ([1, 3, 4, 4], [1, 3, 2, 2], [1, 3, 3, 7], [1, 3, 8, 8]):
            stub.set_input([shape])
            self.assertEqual(
                stub.outputs["y"].shape(), [1, 3, 8, 8], "input {}".format(shape)
            )

    def test_resize_not_larger_rechooses_its_ratio_per_input(self):
        """One ratio across the resized axes, re-chosen for each input."""
        stub = import_model(
            _dynamic_resize_by_sizes_model([7, 8], policy="not_larger", axes=[2, 3])
        )
        for shape, expected in (
            ([1, 3, 2, 4], [1, 3, 4, 8]),
            ([1, 3, 4, 4], [1, 3, 7, 7]),
            ([1, 3, 6, 3], [1, 3, 7, 4]),
        ):
            stub.set_input([shape])
            self.assertEqual(
                stub.outputs["y"].shape(), expected, "input {}".format(shape)
            )

    def test_convtranspose_follows_dynamic_spatial_dimensions(self):
        stub = import_model(_dynamic_convtranspose_model())
        for shape, expected in (
            ([1, 3, 4, 4], [1, 2, 8, 8]),
            ([1, 3, 2, 6], [1, 2, 4, 12]),
            ([1, 3, 7, 5], [1, 2, 14, 10]),
        ):
            stub.set_input([shape])
            self.assertEqual(
                stub.outputs["y"].shape(), expected, "input {}".format(shape)
            )

    def test_rank_three_global_average_pool_tracks_dynamic_width(self):
        model = make_model(
            [helper.make_node("GlobalAveragePool", ["x"], ["y"])],
            [value_info("x", ["batch", 3, "width"])],
            [value_info("y", ["batch", 3, 1])],
        )
        stub = import_model(model)
        self.assertEqual(stub.outputs["y"].shape(), [1, 3, 1])

        stub.set_input([[2, 3, 5]])
        self.assertEqual(stub.outputs["y"].shape(), [2, 3, 1])

    def test_input_shapes_seed_dynamic_geometry_before_construction(self):
        stub = import_model(_dynamic_conv_model(), input_shapes={"x": [1, 3, 5, 5]})
        self.assertEqual(stub.inputs["x"].shape(), [1, 3, 5, 5])
        self.assertEqual(stub.outputs["y"].shape(), [1, 2, 3, 3])

        stub.set_input([[2, 3, 7, 5]])
        self.assertEqual(stub.outputs["y"].shape(), [2, 2, 5, 3])

    def test_input_shapes_reject_unknown_or_fixed_dimensions(self):
        with self.assertRaisesRegex(ValueError, "not graph inputs"):
            import_model(_dynamic_conv_model(), input_shapes={"missing": [1]})
        with self.assertRaises(RuntimeError):
            import_model(_dynamic_conv_model(), input_shapes={"x": [1, 4, 5, 5]})

    def test_set_input_requires_one_shape_per_input(self):
        model = make_model(
            [helper.make_node("Identity", ["x"], ["y"])],
            [value_info("x", [1, 2])],
            [value_info("y", [1, 2])],
        )
        stub = import_model(model)

        with self.assertRaisesRegex(ValueError, "expected 1, got 0"):
            stub.set_input([])
        with self.assertRaisesRegex(ValueError, "expected 1, got 2"):
            stub.set_input([[1, 2], [1, 2]])

    def test_dim_descs_distinguish_dimension_states(self):
        # "batch" and "height" are symbolic, 3 is fixed, the last one is
        # dynamic but anonymous.
        shape = ["batch", 3, "height", None]
        model = make_model(
            [helper.make_node("Identity", ["images"], ["y"])],
            [value_info("images", shape)],
            [value_info("y", shape)],
        )
        stub = import_model(model)
        images = stub.inputs["images"]

        self.assertEqual(
            [(desc.dynamic, desc.name) for desc in images.dim_descs()],
            [(True, "batch"), (False, ""), (True, "height"), (True, "")],
        )
        self.assertEqual(
            [images.is_dim_dynamic(i) for i in range(4)],
            [True, False, True, True],
        )
        self.assertEqual(
            [images.dim_name(i) for i in range(4)],
            ["batch", "", "height", ""],
        )

    def test_static_model_declares_no_dynamic_dim(self):
        model = make_model(
            [helper.make_node("Identity", ["x"], ["y"])],
            [value_info("x", [1, 2])],
            [value_info("y", [1, 2])],
        )
        stub = import_model(model)

        # Nothing was declared, so the shape stays replaceable as a whole.
        self.assertEqual(stub.inputs["x"].dim_descs(), [])
        stub.set_input([[8192, 2]])
        self.assertEqual(stub.getShape("y"), [8192, 2])

    def test_set_input_accepts_dynamic_dims(self):
        model = make_model(
            [helper.make_node("Identity", ["images"], ["y"])],
            [value_info("images", ["batch", 3, "height", "width"])],
            [value_info("y", ["batch", 3, "height", "width"])],
        )
        stub = import_model(model)

        stub.set_input([[4, 3, 224, 224]])
        self.assertEqual(stub.getShape("images"), [4, 3, 224, 224])
        self.assertEqual(stub.getShape("y"), [4, 3, 224, 224])

    def test_set_input_applies_a_change_beside_an_unchanged_shape(self):
        model = make_model(
            [helper.make_node("Add", ["a", "b"], ["y"])],
            [value_info("a", ["batch", 4]), value_info("b", ["batch", 4])],
            [value_info("y", ["batch", 4])],
        )
        stub = import_model(model)

        # Asking for a shape a tensor already carries is skipped as having
        # nothing to do, so a change asked for beside it must still land.
        stub.set_input([[3, 4], [3, 4]])
        self.assertEqual(stub.tensors["y"].shape(), [3, 4])

        stub.set_input([[5, 4], [5, 4]])
        self.assertEqual(stub.tensors["y"].shape(), [5, 4])

        # And the same shape twice over leaves everything as it was.
        stub.set_input([[5, 4], [5, 4]])
        self.assertEqual(stub.tensors["y"].shape(), [5, 4])

    def test_set_input_keeps_the_data_written_before_it(self):
        stub = import_model(_reshape_chain_model([15]))
        stub.set_input([[2, 3, 5]])
        written = np.arange(2 * 15, dtype=np.float32)
        stub.tensors["x"].copyin_float(written.tolist())

        # A shape that is already in place leaves the memory alone, so what was
        # written into it is still there to be run on.
        stub.set_input([[2, 3, 5]])
        stub.run()

        got = np.asarray(stub.tensors["y"].copyout_float(), dtype=np.float32)
        self.assertTrue(np.array_equal(got, written))

    def test_set_input_rejects_fixed_dim_change(self):
        model = make_model(
            [helper.make_node("Identity", ["images"], ["y"])],
            [value_info("images", ["batch", 3, "height", "width"])],
            [value_info("y", ["batch", 3, "height", "width"])],
        )
        stub = import_model(model)

        with self.assertRaisesRegex(
            RuntimeError, 'input "images".*dim 1 is fixed, expected 3 but got 5'
        ):
            stub.set_input([[4, 5, 224, 224]])
        with self.assertRaisesRegex(RuntimeError, "declares rank 4 but got rank 3"):
            stub.set_input([[4, 3, 224]])
        with self.assertRaisesRegex(RuntimeError, "dim 0 must be positive, but got 0"):
            stub.set_input([[0, 3, 224, 224]])

    def test_rejected_later_input_preserves_shapes_and_recovery(self):
        model = make_model(
            [helper.make_node("Add", ["a", "b"], ["y"])],
            [value_info("a", ["batch", 4]), value_info("b", [1, "width"])],
            [value_info("y", ["batch", 4])],
        )
        stub = import_model(model)
        stub.set_input([[2, 4], [1, 4]])
        stub.pin_dims("b", [1])
        a = np.arange(8, dtype=np.float32).reshape(2, 4)
        b = np.arange(4, dtype=np.float32).reshape(1, 4)
        stub.inputs["a"].copyin_numpy(a)
        stub.inputs["b"].copyin_numpy(b)
        stub.run()

        # A later rejection must not leave an earlier input with new metadata
        # but old storage and outputs. It must remain safe to run the old data.
        with self.assertRaisesRegex(RuntimeError, 'input "b".*dim 1 is fixed'):
            stub.set_input([[7, 4], [1, 5]])
        self.assertEqual(stub.inputs["a"].shape(), [2, 4])
        self.assertEqual(stub.inputs["b"].shape(), [1, 4])
        self.assertEqual(stub.outputs["y"].shape(), [2, 4])
        stub.run()
        np.testing.assert_array_equal(stub.outputs["y"].copyout_numpy(), a + b)

        # This request must infer/allocate: a partial earlier mutation would
        # otherwise make the same-shape shortcut silently keep y at [2, 4].
        stub.set_input([[7, 4], [1, 4]])
        a = np.arange(28, dtype=np.float32).reshape(7, 4)
        self.assertEqual(stub.outputs["y"].shape(), [7, 4])
        stub.inputs["a"].copyin_numpy(a)
        stub.inputs["b"].copyin_numpy(b)
        stub.run()
        np.testing.assert_array_equal(stub.outputs["y"].copyout_numpy(), a + b)

    def test_shape_operator_reports_int64_dims(self):
        model = make_model(
            [helper.make_node("Shape", ["x"], ["s"])],
            [value_info("x", ["batch", 3, 5])],
            [value_info("s", [3], TensorProto.INT64)],
        )
        stub = import_model(model)

        # ONNX fixes the output of Shape at int64, whatever the input holds.
        self.assertEqual(stub.tensors["s"].dtype(), TensorProto.INT64)
        self.assertEqual(stub.tensors["s"].shape_value(), [1, 3, 5])

        stub.set_input([[8, 3, 5]])

        self.assertEqual(stub.tensors["s"].shape_value(), [8, 3, 5])

    def test_small_integer_initializer_carries_its_value(self):
        model = make_model(
            [helper.make_node("Gather", ["axes", "index"], ["y"])],
            [value_info("index", [1], TensorProto.INT64)],
            [value_info("y", [1], TensorProto.INT64)],
            [initializer("axes", [0, 2], np.int64)],
        )
        stub = import_model(model)

        self.assertEqual(stub.tensors["axes"].shape_value(), [0, 2])

    def test_shape_subgraph_works_out_its_dimensions(self):
        model = make_model(
            [
                helper.make_node("Shape", ["x"], ["s"]),
                helper.make_node("Gather", ["s", "first"], ["g"], axis=0),
                helper.make_node("Unsqueeze", ["g", "zero"], ["u"]),
                helper.make_node("Concat", ["u", "tail"], ["c"], axis=0),
            ],
            [value_info("x", ["batch", 3, 5])],
            [value_info("c", [2], TensorProto.INT64)],
            [
                initializer("first", 0, np.int64),
                initializer("zero", [0], np.int64),
                initializer("tail", [15], np.int64),
            ],
        )
        stub = import_model(model)

        values = lambda: [stub.tensors[n].shape_value() for n in "sguc"]
        self.assertEqual(values(), [[1, 3, 5], [1], [1], [1, 15]])

        stub.set_input([[8, 3, 5]])

        # Every step of the chain follows the new shape of the input.
        self.assertEqual(values(), [[8, 3, 5], [8], [8], [8, 15]])

    def test_reshape_reads_its_target_from_a_shape_subgraph(self):
        model = _reshape_chain_model([15])
        stub = import_model(model)

        self.assertEqual(stub.tensors["y"].shape(), [1, 15])

        # A shape the graph has already seen has to give the same answer the
        # second time, which is the one way a stale cached value would show.
        for dims, expected in ([4, 3, 5], [4, 15]), ([7, 3, 5], [7, 15]), (
            [4, 3, 5],
            [4, 15],
        ):
            stub.set_input([dims])
            self.assertEqual(stub.tensors["y"].shape(), expected)

    def test_shape_subgraph_runs_and_reshapes_the_data(self):
        stub = import_model(_reshape_chain_model([15]))

        # A shape the graph has already run has to give the same answer the
        # second time, which is the one way a stale cached value would show.
        for dims in [4, 3, 5], [7, 3, 5], [4, 3, 5]:
            stub.set_input([dims])
            data = np.arange(np.prod(dims), dtype=np.float32).reshape(dims)
            stub.tensors["x"].copyin_float(data.flatten().tolist())

            stub.run()

            # Comparing the numbers matters as much as the shape here: a chain
            # that worked out the wrong target would still hand back a tensor
            # of a plausible shape holding the wrong elements.
            got = np.asarray(stub.tensors["y"].copyout_float(), dtype=np.float32)
            self.assertEqual(stub.tensors["y"].shape(), [dims[0], 15])
            self.assertTrue(
                np.array_equal(got.reshape(dims[0], 15), data.reshape(dims[0], 15))
            )

    def test_reshape_resolves_minus_one_from_a_shape_subgraph(self):
        model = _reshape_chain_model([-1])
        stub = import_model(model)

        # The target says [batch, -1], so the rest of the elements land in the
        # second dimension.
        self.assertEqual(stub.tensors["y"].shape(), [1, 15])

        stub.set_input([[6, 3, 5]])

        self.assertEqual(stub.tensors["y"].shape(), [6, 15])

    def test_static_reshape_keeps_reading_a_constant_target(self):
        model = make_model(
            [helper.make_node("Reshape", ["x", "target"], ["y"])],
            [value_info("x", [2, 3, 4])],
            [value_info("y", [6, 4])],
            [initializer("target", [6, 4], np.int64)],
        )
        stub = import_model(model)

        self.assertEqual(stub.tensors["y"].shape(), [6, 4])
        # A constant target stays an attribute, so the operator keeps the one
        # input it has always had rather than gaining an edge.
        reshape = _only_reshape(stub)
        self.assertEqual(len(reshape.inputs()), 1)

    def test_dynamic_reshape_takes_its_target_as_an_edge(self):
        stub = import_model(_reshape_chain_model([15]))

        reshape = _only_reshape(stub)
        self.assertEqual(len(reshape.inputs()), 2)

    def test_reshape_rejects_a_target_holding_real_data(self):
        model = make_model(
            [
                helper.make_node("Gather", ["table", "index"], ["target"], axis=0),
                helper.make_node("Reshape", ["x", "target"], ["y"]),
            ],
            [
                value_info("x", [2, 3]),
                value_info("index", [2], TensorProto.INT64),
            ],
            [value_info("y", [2, 3])],
            [initializer("table", [[1, 2], [3, 4], [5, 6]], np.int64)],
        )

        # The target is gathered from a table of real numbers, so no shape can
        # be worked out from it and the graph has to say so rather than guess.
        with self.assertRaisesRegex(RuntimeError, "holds data rather than dimensions"):
            import_model(model)

    def test_a_shape_subgraph_feeds_real_work(self):
        weight = np.arange(12, dtype=np.float32).reshape(4, 3) / 8 - 0.5
        bias = np.array([0.25, -0.5, 0.125], dtype=np.float32)
        model = make_model(
            [
                # Flatten by a batch size read off the input at run time.
                helper.make_node("Shape", ["x"], ["s"]),
                helper.make_node("Gather", ["s", "first"], ["batch"], axis=0),
                helper.make_node("Unsqueeze", ["batch", "zero"], ["batch_1d"]),
                helper.make_node("Concat", ["batch_1d", "four"], ["flat"], axis=0),
                helper.make_node("Reshape", ["x", "flat"], ["rows"]),
                helper.make_node("MatMul", ["rows", "weight"], ["hidden"]),
                helper.make_node("Add", ["hidden", "bias"], ["biased"]),
                helper.make_node("Relu", ["biased"], ["y"]),
            ],
            [value_info("x", ["batch", 2, 2])],
            [value_info("y", ["batch", 3])],
            [
                initializer("first", 0, np.int64),
                initializer("zero", [0], np.int64),
                initializer("four", [4], np.int64),
                initializer("weight", weight),
                initializer("bias", bias),
            ],
        )
        stub = import_model(model)

        # 3 twice running takes the unchanged-shape path, and 5 -> 1 -> 5
        # shrinks then grows back, where a buffer left over from the larger
        # shape would still read as the right size.
        for batch in 3, 3, 5, 1, 5:
            x = np.arange(batch * 4, dtype=np.float32).reshape(batch, 2, 2) / 4 - 1
            stub.set_input([[batch, 2, 2]])
            stub.inputs["x"].copyin_numpy(x)

            stub.run()

            got = np.asarray(stub.outputs["y"].copyout_float(), dtype=np.float32)
            self.assertEqual(stub.outputs["y"].shape(), [batch, 3])
            expected = np.maximum(x.reshape(batch, 4) @ weight + bias, 0)
            np.testing.assert_allclose(got.reshape(batch, 3), expected, atol=1e-6)

    def test_two_spatial_dimensions_move_independently(self):
        """Height and width are both dynamic, and the chain reads each of them.

        Reading only the batch leaves the interesting cases untested: here the
        two numbers the chain carries come off an axis a pooling has already
        halved, they move separately from one another, and the tensor cannot be
        put back to four dimensions without both of them.
        """
        weight = (np.arange(4, dtype=np.float32).reshape(2, 2, 1, 1) / 4) - 0.5
        model = make_model(
            [
                helper.make_node(
                    "MaxPool", ["x"], ["p"], kernel_shape=[2, 2], strides=[2, 2]
                ),
                helper.make_node("Conv", ["p", "weight"], ["c"], pads=[0, 0, 0, 0]),
                helper.make_node("Relu", ["c"], ["r"]),
                # Take the batch, the pooled height and the pooled width out of
                # the shape one at a time.
                helper.make_node("Shape", ["r"], ["s"]),
                helper.make_node("Gather", ["s", "i0"], ["n"], axis=0),
                helper.make_node("Gather", ["s", "i2"], ["h"], axis=0),
                helper.make_node("Gather", ["s", "i3"], ["w"], axis=0),
                helper.make_node("Unsqueeze", ["n", "zero"], ["n1"]),
                helper.make_node("Unsqueeze", ["h", "zero"], ["h1"]),
                helper.make_node("Unsqueeze", ["w", "zero"], ["w1"]),
                # Flatten both spatial axes into one whose length only this
                # chain knows, then put the two of them back.
                helper.make_node("Concat", ["n1", "two", "minus1"], ["flat"], axis=0),
                helper.make_node("Reshape", ["r", "flat"], ["rows"]),
                helper.make_node("Softmax", ["rows"], ["soft"], axis=-1),
                helper.make_node("Concat", ["n1", "two", "h1", "w1"], ["back"], axis=0),
                helper.make_node("Reshape", ["soft", "back"], ["y"]),
            ],
            [value_info("x", ["batch", 2, "height", "width"])],
            [value_info("y", ["batch", 2, "oh", "ow"])],
            [
                initializer("weight", weight),
                initializer("i0", 0, np.int64),
                initializer("i2", 2, np.int64),
                initializer("i3", 3, np.int64),
                initializer("zero", [0], np.int64),
                initializer("two", [2], np.int64),
                initializer("minus1", [-1], np.int64),
            ],
        )
        stub = import_model(model)

        rng = np.random.default_rng(0)
        # A repeat takes the unchanged-shape path. Then height and width move
        # one at a time, and the last pair shrinks below an earlier one, where
        # a buffer left from the larger shape would still read as big enough.
        for n, h, w in [(1, 8, 8), (1, 8, 8), (1, 8, 12), (2, 12, 8), (1, 4, 4)]:
            x = rng.standard_normal((n, 2, h, w)).astype(np.float32)
            stub.set_input([[n, 2, h, w]])
            stub.inputs["x"].copyin_numpy(x)

            stub.run()

            ph, pw = h // 2, w // 2
            # MaxPool 2x2 stride 2 halves both spatial dimensions, then a 1x1
            # convolution mixes the channels and leaves them alone.
            pooled = x.reshape(n, 2, ph, 2, pw, 2).max(axis=(3, 5))
            mixed = np.einsum("nchw,fc->nfhw", pooled, weight[:, :, 0, 0])
            flat = np.maximum(mixed, 0).reshape(n, 2, -1)
            shifted = flat - flat.max(axis=-1, keepdims=True)
            soft = np.exp(shifted) / np.exp(shifted).sum(axis=-1, keepdims=True)

            self.assertEqual(stub.outputs["y"].shape(), [n, 2, ph, pw])
            got = np.asarray(stub.outputs["y"].copyout_float(), dtype=np.float32)
            np.testing.assert_allclose(
                got.reshape(n, 2, ph, pw), soft.reshape(n, 2, ph, pw), atol=1e-6
            )

    def test_data_operators_work_out_no_dimensions(self):
        model = make_model(
            [helper.make_node("Gather", ["table", "index"], ["y"], axis=0)],
            [value_info("index", [1], TensorProto.INT64)],
            [value_info("y", [1, 2])],
            [initializer("table", [[1.0, 2.0], [3.0, 4.0]], np.float32)],
        )
        stub = import_model(model)

        # Gathering rows out of a float table says nothing about any shape.
        self.assertIsNone(stub.tensors["y"].shape_value())

    def test_unusable_initializers_carry_no_value(self):
        model = make_model(
            [helper.make_node("MatMul", ["x", "weight"], ["y"])],
            [value_info("x", [1, 2])],
            [value_info("y", [1, 2])],
            [initializer("weight", [[1.0, 0.0], [0.0, 1.0]], np.float32)],
        )
        stub = import_model(model)

        # A float tensor cannot describe a shape, and a matrix is the wrong rank.
        self.assertIsNone(stub.tensors["weight"].shape_value())

    def test_initializer_is_restored_after_reallocation(self):
        weight = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        model = make_model(
            [helper.make_node("MatMul", ["x", "weight"], ["y"])],
            [value_info("x", [1, 2])],
            [value_info("y", [1, 2])],
            [initializer("weight", weight)],
        )
        for use_naive_allocator in (False, True):
            with self.subTest(use_naive_allocator=use_naive_allocator):
                stub = import_model(model, use_naive_allocator=use_naive_allocator)
                for batch in (2, 1, 8192, 3):
                    x = np.arange(batch * 2, dtype=np.float32).reshape(batch, 2)
                    stub.set_input([[batch, 2]])
                    stub.inputs["x"].copyin_numpy(x)
                    stub.run()
                    actual = np.asarray(stub.outputs["y"].copyout_float()).reshape(
                        batch, 2
                    )
                    np.testing.assert_allclose(actual, x @ weight)
                if not use_naive_allocator:
                    stub.trim_memory()
                    stub.run()
                    actual = np.asarray(stub.outputs["y"].copyout_float()).reshape(
                        3, 2
                    )
                    np.testing.assert_allclose(actual, x @ weight)

    def test_initializers_are_not_written_again_for_a_new_shape(self):
        """A shape change leaves the weights where they are, so they stand.

        Weight storage is handed out once and never moved for as long as the
        graph lives, so reading every initializer out of the model and writing
        it in again on each shape produces what is already there. On a model
        whose weights are large it is the greater part of what a shape change
        costs, so it is the work itself that has to go, not merely its result
        that gets cached.
        """
        weight = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        bias = np.array([0.5, -0.5], dtype=np.float32)
        model = make_model(
            [
                helper.make_node("MatMul", ["x", "weight"], ["m"]),
                helper.make_node("Add", ["m", "bias"], ["y"]),
            ],
            [value_info("x", ["batch", 2])],
            [value_info("y", ["batch", 2])],
            [initializer("weight", weight), initializer("bias", bias)],
        )
        for use_naive_allocator in (False, True):
            with self.subTest(use_naive_allocator=use_naive_allocator):
                stub = import_model(model, use_naive_allocator=use_naive_allocator)
                # Importing the model copied them in; what follows is the part
                # that has nothing left to do.
                with patch.object(
                    onnx_frontend, "to_array", wraps=onnx_frontend.to_array
                ) as to_array:
                    for batch in (2, 1, 5, 3, 5):
                        x = np.arange(batch * 2, dtype=np.float32).reshape(batch, 2)
                        stub.set_input([[batch, 2]])
                        stub.inputs["x"].copyin_numpy(x)
                        stub.run()
                        actual = np.asarray(stub.outputs["y"].copyout_float()).reshape(
                            batch, 2
                        )
                        np.testing.assert_allclose(actual, x @ weight + bias)
                self.assertEqual(to_array.call_count, 0)


class TestStaticOnnxInputs(unittest.TestCase):
    def test_empty_optional_inputs_are_supported(self):
        cases = []

        max_value = initializer("max", np.array(1.0, dtype=np.float32))
        cases.append(
            make_model(
                [helper.make_node("Clip", ["x", "", "max"], ["y"])],
                [value_info("x", [2])],
                [value_info("y", [2])],
                [max_value],
            )
        )

        cases.append(
            make_model(
                [helper.make_node("Squeeze", ["x", ""], ["y"])],
                [value_info("x", [1, 2])],
                [value_info("y", [2])],
            )
        )

        cases.append(
            make_model(
                [helper.make_node("Split", ["x", ""], ["a", "b"], axis=0)],
                [value_info("x", [4])],
                [value_info("a", [2]), value_info("b", [2])],
            )
        )

        starts = initializer("starts", [0], np.int64)
        ends = initializer("ends", [2], np.int64)
        cases.append(
            make_model(
                [helper.make_node("Slice", ["x", "starts", "ends", "", ""], ["y"])],
                [value_info("x", [4])],
                [value_info("y", [2])],
                [starts, ends],
            )
        )

        pads = initializer("pads", [1, 1], np.int64)
        axes = initializer("axes", [1], np.int64)
        cases.append(
            make_model(
                [helper.make_node("Pad", ["x", "pads", "", "axes"], ["y"])],
                [value_info("x", [1, 2])],
                [value_info("y", [1, 4])],
                [pads, axes],
            )
        )

        cases.append(
            make_model(
                [helper.make_node("ReduceSum", ["x", ""], ["y"], keepdims=0)],
                [value_info("x", [2])],
                [value_info("y", [])],
            )
        )

        for model in cases:
            with self.subTest(op=model.graph.node[0].op_type):
                import_model(model)

    def test_required_static_input_rejects_runtime_tensor(self):
        model = make_model(
            [helper.make_node("Clip", ["x", "", "max"], ["y"])],
            [value_info("x", [2]), value_info("max", [], TensorProto.FLOAT)],
            [value_info("y", [2])],
        )

        with self.assertRaisesRegex(
            ValueError, r'Clip input 2 \("max"\) must be constant'
        ):
            import_model(model)

    def test_unsqueeze_without_axes_is_explicitly_rejected(self):
        model = make_model(
            [helper.make_node("Unsqueeze", ["x", ""], ["y"])],
            [value_info("x", [2])],
            [value_info("y", [1, 2])],
            check=False,
        )

        with self.assertRaisesRegex(ValueError, "Unsqueeze requires constant axes"):
            import_model(model)

    def test_pad_rejects_unsupported_mode_and_value(self):
        pads = initializer("pads", [1, 1], np.int64)
        reflect = make_model(
            [helper.make_node("Pad", ["x", "pads"], ["y"], mode="reflect")],
            [value_info("x", [2])],
            [value_info("y", [4])],
            [pads],
        )
        with self.assertRaisesRegex(NotImplementedError, "Pad mode"):
            import_model(reflect)

        value = initializer("value", np.array(1.0, dtype=np.float32))
        nonzero = make_model(
            [helper.make_node("Pad", ["x", "pads", "value"], ["y"])],
            [value_info("x", [2])],
            [value_info("y", [4])],
            [pads, value],
        )
        with self.assertRaisesRegex(NotImplementedError, "value of zero"):
            import_model(nonzero)

    def test_dropout_inference_and_unsupported_features(self):
        inference_training = initializer(
            "training", np.array(False, dtype=np.bool_)
        )
        inference_model = make_model(
            [helper.make_node("Dropout", ["x", "ratio", "training"], ["y"])],
            [
                value_info("x", [2]),
                value_info("ratio", [], TensorProto.FLOAT),
            ],
            [value_info("y", [2])],
            [inference_training],
        )
        import_model(inference_model)

        training = initializer("training", np.array(True, dtype=np.bool_))
        training_model = make_model(
            [helper.make_node("Dropout", ["x", "", "training"], ["y"])],
            [value_info("x", [2])],
            [value_info("y", [2])],
            [training],
        )
        with self.assertRaisesRegex(NotImplementedError, "training mode"):
            import_model(training_model)

        mask_model = make_model(
            [helper.make_node("Dropout", ["x"], ["y", "mask"])],
            [value_info("x", [2])],
            [value_info("y", [2]), value_info("mask", [2], TensorProto.BOOL)],
        )
        with self.assertRaisesRegex(NotImplementedError, "mask output"):
            import_model(mask_model)


class TestOnnxStubExport(unittest.TestCase):
    def assert_valid_export(self, model):
        checker.check_model(model)
        return model.graph.node[0]

    def test_conv_transpose_preserves_attributes(self):
        weight = np.ones((1, 1, 3, 3), dtype=np.float32)
        model = make_model(
            [
                helper.make_node(
                    "ConvTranspose",
                    ["x", "weight"],
                    ["y"],
                    pads=[1, 1, 1, 1],
                    strides=[2, 3],
                    dilations=[1, 2],
                    output_padding=[1, 2],
                )
            ],
            [value_info("x", [1, 1, 3, 3])],
            [value_info("y", [1, 1, 6, 11])],
            [initializer("weight", weight)],
        )
        node = self.assert_valid_export(import_model(model).to_onnx("export"))

        self.assertEqual(node_attribute(node, "pads"), [1, 1, 1, 1])
        self.assertEqual(node_attribute(node, "strides"), [2, 3])
        self.assertEqual(node_attribute(node, "dilations"), [1, 2])
        self.assertEqual(node_attribute(node, "output_padding"), [1, 2])

    def test_softmax_preserves_axis(self):
        model = make_model(
            [helper.make_node("Softmax", ["x"], ["y"], axis=1)],
            [value_info("x", [2, 3, 4])],
            [value_info("y", [2, 3, 4])],
        )
        node = self.assert_valid_export(import_model(model).to_onnx("export"))

        self.assertEqual(node_attribute(node, "axis"), 1)

    def test_split_exports_unequal_output_sizes(self):
        split = initializer("split", [1, 3], np.int64)
        model = make_model(
            [helper.make_node("Split", ["x", "split"], ["a", "b"], axis=1)],
            [value_info("x", [2, 4])],
            [value_info("a", [2, 1]), value_info("b", [2, 3])],
            [split],
        )
        exported = import_model(model).to_onnx("export")
        node = self.assert_valid_export(exported)
        exported_split = next(
            item for item in exported.graph.initializer if item.name == node.input[1]
        )

        self.assertEqual(numpy_helper.to_array(exported_split).tolist(), [1, 3])

    def test_expand_exports_shape_as_int64_input(self):
        shape = initializer("shape", [2, 3], np.int64)
        model = make_model(
            [helper.make_node("Expand", ["x", "shape"], ["y"])],
            [value_info("x", [1, 3])],
            [value_info("y", [2, 3])],
            [shape],
        )
        exported = import_model(model).to_onnx("export")
        node = self.assert_valid_export(exported)
        exported_shape = next(
            item for item in exported.graph.initializer if item.name == node.input[1]
        )

        self.assertEqual(len(node.input), 2)
        self.assertFalse(any(attr.name == "shape" for attr in node.attribute))
        self.assertEqual(exported_shape.data_type, TensorProto.INT64)
        self.assertEqual(numpy_helper.to_array(exported_shape).tolist(), [2, 3])

    def test_clip_exports_optional_inputs_and_bound_dtype(self):
        max_value = initializer("max", np.array(1.0, dtype=np.float16))
        model = make_model(
            [helper.make_node("Clip", ["x", "", "max"], ["y"])],
            [value_info("x", [2], TensorProto.FLOAT16)],
            [value_info("y", [2], TensorProto.FLOAT16)],
            [max_value],
        )
        exported = import_model(model).to_onnx("export")
        node = self.assert_valid_export(exported)
        exported_max = next(
            item for item in exported.graph.initializer if item.name == node.input[2]
        )

        self.assertEqual(node.input[1], "")
        self.assertEqual(exported_max.data_type, TensorProto.FLOAT16)
        self.assertEqual(numpy_helper.to_array(exported_max), np.float16(1.0))

    def test_lrn_exports_named_attributes(self):
        model = make_model(
            [
                helper.make_node(
                    "LRN",
                    ["x"],
                    ["y"],
                    alpha=0.001,
                    beta=0.5,
                    bias=2.0,
                    size=3,
                )
            ],
            [value_info("x", [1, 3, 2, 2])],
            [value_info("y", [1, 3, 2, 2])],
        )
        node = self.assert_valid_export(import_model(model).to_onnx("export"))

        self.assertAlmostEqual(node_attribute(node, "alpha"), 0.001)
        self.assertAlmostEqual(node_attribute(node, "beta"), 0.5)
        self.assertAlmostEqual(node_attribute(node, "bias"), 2.0)
        self.assertEqual(node_attribute(node, "size"), 3)

    def test_shape_subgraph_survives_a_round_trip(self):
        stub = import_model(_reshape_chain_model([15]))

        exported = stub.to_onnx("roundtrip")
        checker.check_model(exported)

        self.assertEqual(
            [node.op_type for node in exported.graph.node],
            ["Shape", "Gather", "Unsqueeze", "Concat", "Reshape"],
        )
        # A Reshape already holding its target as an edge must not have a
        # second one spelled out as a constant beside it.
        reshape = next(
            node for node in exported.graph.node if node.op_type == "Reshape"
        )
        self.assertEqual(len(reshape.input), 2)

        # The exported model has to keep working out its own dimensions, which
        # is what makes the round trip meaningful rather than merely valid.
        name = exported.graph.output[0].name
        again = import_model(exported)
        self.assertEqual(again.tensors[name].shape(), [1, 15])

        again.set_input([[6, 3, 5]])

        self.assertEqual(again.tensors[name].shape(), [6, 15])

    def test_repeated_export_does_not_mutate_initializers(self):
        weight = initializer(
            "weight", np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        )
        model = make_model(
            [helper.make_node("MatMul", ["x", "weight"], ["y"])],
            [value_info("x", [1, 2])],
            [value_info("y", [1, 2])],
            [weight],
        )
        stub = import_model(model)
        internal_before = {
            fuid: tensor.SerializeToString()
            for fuid, tensor in stub.initializer.items()
        }

        first = stub.to_onnx("export")
        second = stub.to_onnx("export")

        checker.check_model(first)
        checker.check_model(second)
        self.assertEqual(first.SerializeToString(), second.SerializeToString())
        self.assertEqual(
            {
                fuid: tensor.SerializeToString()
                for fuid, tensor in stub.initializer.items()
            },
            internal_before,
        )


class TestOnnxStubCuda(unittest.TestCase):
    @unittest.skipUnless(hasattr(backend, "cuda_runtime"), "CUDA backend not built")
    def test_dynamic_reallocation_restores_initializer(self):
        weight = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        model = make_model(
            [helper.make_node("MatMul", ["x", "weight"], ["y"])],
            [value_info("x", [1, 2])],
            [value_info("y", [1, 2])],
            [initializer("weight", weight)],
        )
        for use_naive_allocator in (False, True):
            with self.subTest(use_naive_allocator=use_naive_allocator):
                stub = import_model(
                    model,
                    backend.cuda_runtime(),
                    use_naive_allocator=use_naive_allocator,
                )
                for batch in (3, 1, 8192, 2):
                    x = np.arange(batch * 2, dtype=np.float32).reshape(batch, 2)
                    stub.set_input([[batch, 2]])
                    stub.inputs["x"].copyin_numpy(x)
                    stub.run()
                    actual = np.asarray(stub.outputs["y"].copyout_float()).reshape(
                        batch, 2
                    )
                    np.testing.assert_allclose(actual, x @ weight, rtol=1e-5, atol=1e-6)
                if not use_naive_allocator:
                    stub.trim_memory()
                    stub.run()
                    actual = np.asarray(stub.outputs["y"].copyout_float()).reshape(
                        2, 2
                    )
                    np.testing.assert_allclose(actual, x @ weight, rtol=1e-5, atol=1e-6)


class TestShapeSubgraphFold(unittest.TestCase):
    def test_settled_chain_folds_away(self):
        stub = import_model(_settled_chain_model())
        self.assertEqual(stub.handler.operator_count(), 4)

        dropped = stub.fold_shape_subgraph()

        # The gather and the unsqueeze held a settled result, and the `Shape`
        # feeding them was left with nobody to read it. The `Concat` is what
        # the graph was asked for, so it stays and reads the constants.
        self.assertEqual(dropped, 3)
        self.assertEqual(stub.handler.operator_count(), 1)
        stub.run()
        self.assertEqual(stub.outputs["c"].copyout_int64(), [3, 7])

    def test_moving_chain_is_left_alone(self):
        stub = import_model(_moving_chain_model())

        self.assertEqual(stub.fold_shape_subgraph(), 0)
        self.assertEqual(stub.handler.operator_count(), 4)

        # Still a live computation, so it follows the input as it did before.
        self.assertEqual(stub.tensors["c"].shape_value(), [1, 7])
        stub.set_input([[8, 3, 5]])
        self.assertEqual(stub.tensors["c"].shape_value(), [8, 7])

    def test_folding_reports_the_tensors_it_let_go_of(self):
        stub = import_model(_settled_chain_model())
        gone_before = set(stub.handler.folded_away_tensors())

        stub.fold_shape_subgraph()

        gone = set(stub.handler.folded_away_tensors())
        self.assertEqual(gone_before, set())
        # The intermediates of the chain, and the constants that fed an
        # operator, are no longer the graph's to account for. `zero` is not
        # among them: the importer reads Unsqueeze axes into the operator
        # itself, so that tensor never reached the graph's wiring and the fold
        # has nothing to say about it.
        self.assertEqual(
            {name for name, t in stub.tensors.items() if t.fuid() in gone},
            {"s", "g", "middle"},
        )
        # `u` and `tail` are still read by the `Concat` that stayed, and `zero`
        # never reached the graph's wiring at all: the importer reads Unsqueeze
        # axes into the operator itself.
        self.assertEqual(set(stub._initializer_by_name), {"zero", "tail"})

    def test_folded_target_still_reshapes_across_shapes(self):
        stub = import_model(_flatten_chain_model())

        self.assertEqual(stub.fold_shape_subgraph(), 4)
        # Reshape reads the target through a kernel, so the fold has to leave
        # real numbers behind rather than only metadata.
        self.assertEqual(stub.handler.operator_count(), 1)
        self.assertEqual(stub.tensors["target"].copyout_int64(), [-1, 5])

        # A shape the graph has already run has to give the same answer the
        # second time, which is the one way a stale cached value would show.
        for batch in 4, 7, 4:
            dims = [batch, 3, 5]
            stub.set_input([dims])
            x = np.arange(np.prod(dims), dtype=np.float32).reshape(dims)
            stub.inputs["x"].copyin_numpy(x)

            stub.run()

            self.assertEqual(stub.outputs["y"].shape(), [batch * 3, 5])
            actual = np.asarray(stub.outputs["y"].copyout_float()).reshape(batch * 3, 5)
            np.testing.assert_array_equal(actual, x.reshape(batch * 3, 5))

    def test_axis_from_a_constant_node_folds_the_same_way(self):
        stub = import_model(_flatten_chain_model(axis_from_constant=True))

        # An axis an exporter emitted as a `Constant` carries its value just as
        # an initializer does, so the same part of the chain settles.
        self.assertEqual(stub.tensors["last"].shape_value(), [2])
        self.assertEqual(stub.fold_shape_subgraph(), 4)
        self.assertEqual(stub.tensors["target"].copyout_int64(), [-1, 5])

    def test_optimize_folds_the_chain(self):
        stub = import_model(_settled_chain_model())

        stub.optimize()

        self.assertEqual(stub.handler.operator_count(), 1)
        stub.run()
        self.assertEqual(stub.outputs["c"].copyout_int64(), [3, 7])


if __name__ == "__main__":
    unittest.main()
