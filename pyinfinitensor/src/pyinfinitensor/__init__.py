import importlib
import importlib.util
import os
import sys

sys.path.extend(__path__)


def _provider_modules():
    modules = []
    if importlib.util.find_spec("torch") is not None:
        modules.append("torch")
    modules.extend(
        module.strip()
        for module in os.environ.get("INFINIOPS_PROVIDER_MODULES", "").split(",")
        if module.strip()
    )
    return tuple(dict.fromkeys(modules))


def _initialize_torch_npu_fp32():
    """Initialize ACL while reduced-precision compilation is disabled.

    torch_npu consumes the precision mode during its first initialization, so
    changing the public flags only around a later forward call is too late.
    Restore the application's flags immediately after initialization; the
    embedding call applies the same policy again while it executes.
    """
    import torch
    missing = object()
    saved = []
    for owner, name in ((torch.npu.config, "allow_internal_format"),
                        (torch.npu.conv, "allow_hf32"),
                        (torch.npu.matmul, "allow_hf32")):
        saved.append((owner, name, getattr(owner, name, missing), missing))
        setattr(owner, name, False)
    try:
        torch.npu.init()
    except BaseException:
        _restore_precision_flags(saved)
        raise
    return saved


def _restore_precision_flags(saved):
    for owner, name, value, missing in reversed(saved):
        if value is missing:
            try:
                delattr(owner, name)
            except AttributeError:
                pass
        else:
            setattr(owner, name, value)


_npu_precision_state = []
for _module in _provider_modules():
    importlib.import_module(_module)
    if _module == "torch_npu":
        # Import alone does not initialize the NPU allocator and compiler.
        # C++ ATen calls on externally owned tensors bypass Python lazy init.
        _npu_precision_state = _initialize_torch_npu_fp32()


def _preload_infiniops_libs():
    """Load libinfinirt/libinfiniops with RTLD_GLOBAL before the extension
    so its weak CallX references resolve at load time."""
    import ctypes
    for soname in ("libinfinirt.so", "libinfiniops.so"):
        try:
            ctypes.CDLL(soname, mode=ctypes.RTLD_GLOBAL)
        except OSError:
            pass


try:
    _preload_infiniops_libs()
    import backend
finally:
    _restore_precision_flags(_npu_precision_state)

print("import backend: {}".format(backend))
