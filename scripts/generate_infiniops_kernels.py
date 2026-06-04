"""Generate InfiniTensor kernel bridge files for InfiniOps torch operators.

Reads ``scripts/infiniops_op_mapping.yaml`` and generates one ``.cc`` file per
operator.  Each generated kernel bridges from InfiniTensor's ``OpType`` enum
to the corresponding ``infini::ops::<Op>::Call(handle, config, args...)``.

Phase 1 covers unary (1-in, 1-out) and binary (2-in, 1-out) operators only.
Only ops whose InfiniOps base header exists in ``generated/base/<op>.h`` (or
``src/base/<op>.h`` for hand-written bases) are generated.
"""

import argparse
import pathlib
import sys

import yaml

# ---------------------------------------------------------------------------
# Templates
# ---------------------------------------------------------------------------

_UNARY_TEMPLATE = """\
#include "core/kernel.h"
#include "core/tensor.h"
#include "operators/unary.h"
#include "base/{infiniops_name}.h"
#include "torch/{infiniops_name}/{infiniops_name}.h"

// Each device gets its own kernel class so the Operator template is only
// instantiated for that specific device, avoiding linker errors for
// unlinked device types.
#define INFINIOPS_{pascal_upper}_IMPL(kDev, kOpsDev, kItDev, kRegName) \\
    namespace infini {{ \\
    class {pascal}InfiniOpsKernel_##kDev : public KernelWithoutConfig {{ \\
        void compute(const Operator &op, const RuntimeObj *context) const override {{ \\
            auto unaryOp = as<UnaryObj>(op); \\
            auto input = toInfiniOpsTensor(unaryOp->getInputs(0).get()); \\
            auto output = toInfiniOpsTensor(unaryOp->getOutput().get()); \\
            infini::ops::Handle handle = context->makeHandle(); \\
            infini::ops::Operator<infini::ops::{pascal}, kOpsDev, 8> \\
                impl(input, output); \\
            impl.set_handle(handle); \\
            impl.set_config(infini::ops::Config{{}}); \\
            impl.set_stream(handle.stream()); \\
            impl(input, output); \\
        }} \\
    }}; \\
    }} \\
    REGISTER_KERNEL(kItDev, OpType::{op_type}, infini::{pascal}InfiniOpsKernel_##kDev, \\
                    kRegName);

INFINIOPS_{pascal_upper}_IMPL(Cpu, infini::ops::Device::Type::kCpu, Device(Device::Type::kCpu), "{pascal}_InfiniOps_CPU")
#ifdef WITH_NVIDIA
INFINIOPS_{pascal_upper}_IMPL(Nvidia, infini::ops::Device::Type::kNvidia, Device(Device::Type::kNvidia), "{pascal}_InfiniOps_NVIDIA")
#endif

#undef INFINIOPS_{pascal_upper}_IMPL
"""

_BINARY_TEMPLATE = """\
#include "core/kernel.h"
#include "core/tensor.h"
#include "operators/element_wise.h"
#include "base/{infiniops_name}.h"
#include "torch/{infiniops_name}/{infiniops_name}.h"

#define INFINIOPS_{pascal_upper}_IMPL(kDev, kOpsDev, kItDev, kRegName) \\
    namespace infini {{ \\
    class {pascal}InfiniOpsKernel_##kDev : public KernelWithoutConfig {{ \\
        void compute(const Operator &op, const RuntimeObj *context) const override {{ \\
            auto elemOp = as<ElementWiseObj>(op); \\
            auto input = toInfiniOpsTensor(elemOp->getInputs(0).get()); \\
            auto other = toInfiniOpsTensor(elemOp->getInputs(1).get()); \\
            auto output = toInfiniOpsTensor(elemOp->getOutput().get()); \\
            infini::ops::Handle handle = context->makeHandle(); \\
            infini::ops::Operator<infini::ops::{pascal}, kOpsDev, 8> \\
                impl(input, other, output); \\
            impl.set_handle(handle); \\
            impl.set_config(infini::ops::Config{{}}); \\
            impl.set_stream(handle.stream()); \\
            impl(input, other, output); \\
        }} \\
    }}; \\
    }} \\
    REGISTER_KERNEL(kItDev, OpType::{op_type}, infini::{pascal}InfiniOpsKernel_##kDev, \\
                    kRegName);

INFINIOPS_{pascal_upper}_IMPL(Cpu, infini::ops::Device::Type::kCpu, Device(Device::Type::kCpu), "{pascal}_InfiniOps_CPU")
#ifdef WITH_NVIDIA
INFINIOPS_{pascal_upper}_IMPL(Nvidia, infini::ops::Device::Type::kNvidia, Device(Device::Type::kNvidia), "{pascal}_InfiniOps_NVIDIA")
#endif

#undef INFINIOPS_{pascal_upper}_IMPL
"""


def _snake_to_pascal(s: str) -> str:
    return "".join(part.capitalize() for part in s.split("_"))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate InfiniTensor kernel bridges for InfiniOps torch ops"
    )
    parser.add_argument(
        "--mapping",
        type=pathlib.Path,
        default=pathlib.Path(__file__).resolve().parent / "infiniops_op_mapping.yaml",
        help="Path to the op mapping YAML file",
    )
    parser.add_argument(
        "--infiniops-dir",
        type=pathlib.Path,
        required=True,
        help="Path to the InfiniOps source directory (3rd-party/infiniops)",
    )
    parser.add_argument(
        "--output-dir",
        type=pathlib.Path,
        required=True,
        help="Directory to write generated .cc files",
    )
    args = parser.parse_args()

    mapping = yaml.safe_load(args.mapping.read_text())
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    generated_base = args.infiniops_dir / "generated" / "base"
    hand_base = args.infiniops_dir / "src" / "base"
    torch_dir = args.infiniops_dir / "generated" / "torch"

    templates = {
        "unary": _UNARY_TEMPLATE,
        "binary": _BINARY_TEMPLATE,
    }

    generated = []
    skipped = []

    for category, tmpl in templates.items():
        ops = mapping.get(category, {})
        for op_type, infiniops_name in ops.items():
            # Only generate if the InfiniOps base + torch headers exist.
            gen_header = generated_base / f"{infiniops_name}.h"
            hand_header = hand_base / f"{infiniops_name}.h"
            torch_header = torch_dir / infiniops_name / f"{infiniops_name}.h"
            if not gen_header.exists() and not hand_header.exists():
                skipped.append(
                    (
                        op_type,
                        infiniops_name,
                        f"no base header (checked {gen_header} and {hand_header})",
                    )
                )
                continue
            if not torch_header.exists():
                skipped.append(
                    (op_type, infiniops_name, f"no torch header ({torch_header})")
                )
                continue

            pascal = _snake_to_pascal(infiniops_name)
            source = tmpl.format(
                pascal=pascal,
                pascal_upper=pascal.upper(),
                op_type=op_type,
                infiniops_name=infiniops_name,
            )

            out_path = output_dir / f"{infiniops_name}.cc"
            out_path.write_text(source)
            generated.append((op_type, infiniops_name, out_path))

    # Print summary
    print(
        f"generated {len(generated)} kernel bridges in {output_dir}: "
        f"{sorted(infiniops_name for _, infiniops_name, _ in generated)}"
    )

    for op_type, infiniops_name, reason in skipped:
        print(f"  skipped {infiniops_name!r} ({op_type}): {reason}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(main())
