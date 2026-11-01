"""Verify the installed extension and one real matrix operation."""

import argparse
import os


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--backend",
        default="cpu",
        help="cpu, infini (SDK default), or an InfiniRT device name",
    )
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument(
        "--provider-modules", default=os.environ.get("INFINIOPS_PROVIDER_MODULES", "")
    )
    args = parser.parse_args()
    os.environ["INFINIOPS_PROVIDER_MODULES"] = args.provider_modules
    import numpy as np
    from pyinfinitensor import backend

    target = (
        backend.default_infini_device() if args.backend == "infini" else args.backend
    )
    runtime = (
        backend.runtime("cpu", args.device)
        if args.backend == "cpu"
        else backend.InfiniRuntime(target, args.device)
    )
    handler = backend.GraphHandler(runtime)
    a = handler.tensor([2, 3], 1)
    b = handler.tensor([3, 2], 1)
    a.set_input()
    b.set_input()
    y = handler.matmul(
        a, b, None, False, False, None, backend.ActType.Linear, "default"
    )
    y.set_output()
    handler.data_malloc()
    av = np.arange(6, dtype=np.float32).reshape(2, 3)
    bv = np.arange(6, dtype=np.float32).reshape(3, 2)
    a.copyin_numpy(av)
    b.copyin_numpy(bv)
    handler.run()
    np.testing.assert_allclose(y.copyout_numpy(), av @ bv, atol=1e-5, rtol=1e-5)
    print("Installation check passed: {} device {}".format(args.backend, args.device))


if __name__ == "__main__":
    main()
