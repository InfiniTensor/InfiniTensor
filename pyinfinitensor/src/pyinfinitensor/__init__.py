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


for _module in _provider_modules():
    importlib.import_module(_module)


def _preload_infiniops_libs():
    """Load libinfinirt/libinfiniops with RTLD_GLOBAL before the extension
    so its weak CallX references resolve at load time."""
    import ctypes
    for soname in ("libinfinirt.so", "libinfiniops.so"):
        try:
            ctypes.CDLL(soname, mode=ctypes.RTLD_GLOBAL)
        except OSError:
            pass


_preload_infiniops_libs()

import backend

print("import backend: {}".format(backend))
