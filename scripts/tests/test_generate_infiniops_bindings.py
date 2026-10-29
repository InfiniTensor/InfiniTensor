"""Contract tests for consuming the generated public InfiniOps SDK."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

spec = importlib.util.spec_from_file_location(
    "bindings", Path(__file__).resolve().parents[1] / "generate_infiniops_bindings.py"
)
bindings = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bindings)


def sdk(name, arguments):
    return f"""template<> struct detail::OperatorFactory<::infini::ops::{name}> {{
    static void Call(const Handle& h, const Config& c, {arguments});
    static std::unique_ptr<Operator<{name}>> Make(const Config& c, {arguments});
    }};"""


class PublicSdkContractTest(unittest.TestCase):
    def test_new_name_lists_optional_and_enum_need_no_operator_map(self):
        header = sdk("NewOperation", "const std::vector<Tensor>& inputs, "
                     "const std::optional<DataType>& dtype, const int64_t& axis, "
                     "const Tensor& out")
        operations, unsupported = bindings.discover(header)
        self.assertEqual(unsupported, [])
        self.assertEqual(operations[0]["name"], "NewOperation")
        self.assertEqual(operations[0]["parameters"], [
            {"name": "inputs", "type": "std::vector<Tensor>"},
            {"name": "dtype", "type": "std::optional<DataType>"},
            {"name": "axis", "type": "int64_t"},
            {"name": "out", "type": "Tensor"},
        ])

    def test_execution_only_overload_is_not_instantiable(self):
        header = sdk("Example", "const Tensor& input, const Tensor& out")
        header = header.replace("};", "static void Call(const Handle& h, "
                                "const Config& c, const Tensor& out); }; ")
        operations, _ = bindings.discover(header)
        self.assertEqual(len(operations), 1)
        self.assertEqual(len(operations[0]["parameters"]), 2)

    def test_unsupported_types_are_reported_and_incompatible_sdk_fails(self):
        operations, unsupported = bindings.discover(sdk("Example", "const Special& config"))
        self.assertEqual(operations, [])
        self.assertEqual(unsupported[0]["name"], "Example")
        with self.assertRaisesRegex(ValueError, "incompatible SDK"):
            bindings.discover("// No generated factories installed")

    def test_plain_and_namespaced_factories_are_all_accounted_for(self):
        names = ["Add", "linalg::Inv", "special::Erf", "fft::Fft", "internal::Softmax"]
        header = "\n".join(sdk(name, "const Tensor& input, const Tensor& out") for name in names)
        operations, unsupported = bindings.discover(header)
        self.assertEqual([op["name"] for op in operations], names)
        self.assertEqual(unsupported, [])
        # A partially recognizable header must never silently drop a factory.
        with self.assertRaisesRegex(ValueError, "Incomplete"):
            bindings.discover(header + "struct detail::OperatorFactory<::infini::ops::Broken> {")

    def test_every_unbound_factory_is_reported(self):
        header = sdk("special::Example", "const Special& config")
        header += "struct detail::OperatorFactory<::infini::ops::Empty> {};"
        operations, unsupported = bindings.discover(header)
        self.assertEqual(operations, [])
        self.assertEqual({op["name"] for op in unsupported}, {"special::Example", "Empty"})
        self.assertTrue(all(op["reason"] for op in unsupported))

    def test_generated_consumer_uses_public_make_and_is_reproducible(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            header, output, manifest = (root / name for name in ("sdk.h", "bindings.cc", "bindings.json"))
            header.write_text(sdk("special::NewOperation", "const Tensor& input, const Tensor& out"))
            bindings.generate(header, output, manifest)
            first = output.read_bytes(), manifest.read_bytes()
            bindings.generate(header, output, manifest)
            self.assertEqual(first, (output.read_bytes(), manifest.read_bytes()))
            self.assertIn("prepareTyped<::infini::ops::special::NewOperation>", output.read_text())
            self.assertNotIn("generated_dispatch", output.read_text())
            self.assertEqual(len(json.loads(manifest.read_text())["sdk_sha256"]), 64)
            self.assertEqual(json.loads(manifest.read_text())["discovered_factories"], ["special::NewOperation"])


if __name__ == "__main__":
    unittest.main()
