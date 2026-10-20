"""CPU shape-subgraph lowering and conservative partial evaluation.

Only metadata-derived shape programs are accepted. Data-dependent shapes (for
example NonZero) require a different scheduling contract and are rejected.
The resulting small tensors are materialized before C++ graph shape inference.
"""

import numpy as np
from onnx import helper, numpy_helper


def dimension_schema(value_info):
    shape = value_info.type.tensor_type
    if not shape.HasField("shape"):
        raise ValueError("Unknown input rank is not supported: " + value_info.name)
    result = []
    for dim in shape.shape.dim:
        kind = dim.WhichOneof("value")
        result.append(dim.dim_value if kind == "dim_value" else
                      dim.dim_param if kind == "dim_param" else None)
    return tuple(result)


def validate_shapes(schemas, shapes):
    if len(schemas) != len(shapes):
        raise ValueError("inputShapes must contain one shape per model input; "
                         "expected {}, got {}".format(len(schemas), len(shapes)))
    symbols = {}
    result = {}
    for (name, schema), actual in zip(schemas.items(), shapes):
        if len(schema) != len(actual):
            raise ValueError("{}: expected rank {}, got {}".format(
                name, len(schema), len(actual)))
        dims = []
        for axis, (declared, value) in enumerate(zip(schema, actual)):
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
                raise ValueError("{} axis {}: dimension must be an integer".format(name, axis))
            value = int(value)
            if value <= 0 or value > np.iinfo(np.int32).max:
                raise ValueError("{} axis {}: only positive int32 dimensions supported".format(name, axis))
            if isinstance(declared, int) and declared != value:
                raise ValueError("{} axis {}: fixed dimension {}, got {}".format(
                    name, axis, declared, value))
            if isinstance(declared, str) and declared:
                if declared in symbols and symbols[declared] != value:
                    raise ValueError("Symbol {} has inconsistent dimensions".format(declared))
                symbols[declared] = value
            dims.append(value)
        result[name] = dims
    return result


class ShapeProgram:
    supported = {"Shape", "Gather", "Unsqueeze", "Squeeze", "Concat", "Cast", "Constant", "Identity"}

    def __init__(self, model, ordered_nodes, schemas, optimize=True):
        self.schemas = schemas
        self.constants = {t.name: numpy_helper.to_array(t).copy()
                          for t in model.graph.initializer}
        self.weight_shapes = {t.name: list(t.dims) for t in model.graph.initializer}
        producers = {out: node for node in ordered_nodes for out in node.output}
        selected = set()

        def visit(name):
            if name in self.constants or name in selected:
                return
            node = producers.get(name)
            if node is None or node.op_type not in self.supported or node.domain:
                raise ValueError("Unsupported runtime shape dependency: " + name)
            if len(node.output) != 1:
                raise ValueError("Shape program requires single-output operators")
            selected.add(name)
            if node.op_type == "Shape":
                source = node.input[0]
                if source not in schemas and source not in self.weight_shapes:
                    raise ValueError("Shape currently requires a graph input or initializer: " + source)
            else:
                for inp in node.input:
                    if inp:
                        visit(inp)

        self.targets = set()
        for node in ordered_nodes:
            if node.op_type == "Reshape" and len(node.input) > 1:
                name = node.input[1]
                if name not in self.constants:
                    self.targets.add(name)
                    visit(name)
        # Also lower metadata branches exposed as graph outputs or left unused
        # by a target. They must not fall back to absent native integer kernels.
        for node in ordered_nodes:
            if (node.op_type == "Shape" and not node.domain and
                    node.input[0] in set(schemas) | set(self.weight_shapes)):
                visit(node.output[0])
            elif (node.op_type in self.supported and not node.domain and
                  any(name in selected for name in node.input) and
                  all(not name or name in selected or name in self.constants or
                      (name in producers and producers[name].op_type == "Constant")
                      for name in node.input)):
                visit(node.output[0])
        self.nodes = [node for node in ordered_nodes if node.output[0] in selected]
        self.output_names = selected
        self.folded = {}
        abstract = dict(self.constants)
        abstract_shapes = {name: [d if isinstance(d, int) else None for d in schema]
                           for name, schema in schemas.items()}
        abstract_shapes.update(self.weight_shapes)
        sample_shapes = {name: [1 if d is None else d for d in dims]
                         for name, dims in abstract_shapes.items()}
        sample_values = dict(self.constants)
        for node in self.nodes:
            value = self._evaluate(node, abstract, abstract_shapes, abstract=True)
            abstract[node.output[0]] = value
            sample = self._evaluate(node, sample_values, sample_shapes)
            sample_values[node.output[0]] = sample
            if optimize and not any(v is None for v in value.flat):
                # Object arrays encode partially known dimensions only while
                # compiling. Preserve the concrete ONNX dtype when folding.
                value = value.astype(sample.dtype)
                self.folded[node.output[0]] = value.copy()
        self.stats = {"nodes_before": len(self.nodes),
                      "nodes_folded": len(self.folded),
                      "runtime_nodes": len(self.nodes) - len(self.folded)}

    @staticmethod
    def _evaluate(node, values, shapes, abstract=False):
        attrs = {a.name: helper.get_attribute_value(a) for a in node.attribute}
        op = node.op_type
        # Shape reads dimensions, not the data tensor's contents.
        if op == "Shape":
            dims = shapes[node.input[0]]
            return np.asarray(dims, dtype=object if abstract else np.int64)[
                slice(attrs.get("start"), attrs.get("end"))]
        if op == "Constant":
            if "value" not in attrs:
                raise ValueError("Shape Constant requires tensor-valued 'value'")
            return numpy_helper.to_array(attrs["value"]).copy()
        args = [values[name] for name in node.input if name]
        if op == "Identity":
            return args[0].copy()
        if op == "Gather":
            if args[1].dtype not in (np.dtype("int32"), np.dtype("int64")):
                raise ValueError("Gather indices must be int32 or int64")
            return np.asarray(np.take(args[0], args[1], axis=attrs.get("axis", 0)))
        if op in ("Unsqueeze", "Squeeze"):
            axes = args[1] if len(args) > 1 else attrs.get("axes")
            if axes is not None:
                if any(x is None for x in np.asarray(axes).flat):
                    raise ValueError("Shape program axes must be constant")
                if not abstract and np.asarray(axes).dtype != np.int64:
                    raise ValueError("Shape program axes must be int64")
                axes = tuple(int(a) for a in np.asarray(axes).reshape(-1))
            if op == "Unsqueeze":
                if axes is None:
                    raise ValueError("Unsqueeze requires axes")
                return np.expand_dims(args[0], axes)
            return np.squeeze(args[0], axis=axes)
        if op == "Concat":
            if len({a.dtype for a in args}) > 1 and not abstract:
                raise ValueError("Concat inputs must have the same dtype")
            return np.concatenate(args, axis=attrs["axis"])
        if op == "Cast":
            dtype = helper.tensor_dtype_to_np_dtype(attrs["to"])
            if dtype not in (np.dtype("int32"), np.dtype("int64")):
                raise ValueError("Shape Cast supports int32/int64 destinations only")
            if abstract and any(x is None for x in args[0].flat):
                return args[0].copy()
            return args[0].astype(dtype)
        raise ValueError("Unsupported shape operator: " + op)

    def evaluate(self, input_shapes):
        shapes = dict(self.weight_shapes, **input_shapes)
        values = dict(self.constants)
        for node in self.nodes:
            name = node.output[0]
            try:
                values[name] = (self.folded[name].copy() if name in self.folded else
                                self._evaluate(node, values, shapes))
            except (ValueError, IndexError, TypeError) as error:
                raise ValueError("Shape node {} ({}): {}".format(
                    node.name, node.op_type, error)) from error
        for name in self.targets:
            value = values[name]
            if value.dtype != np.int64 or value.ndim != 1:
                raise ValueError("Reshape target {} must be a rank-1 int64 tensor".format(name))
        return {node.output[0]: np.array(values[node.output[0]], copy=True, order="C")
                for node in self.nodes}
