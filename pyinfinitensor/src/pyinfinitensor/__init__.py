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

import backend

print("import backend: {}".format(backend))
