import sys
from pathlib import Path

sys.path.append(str(Path(str(__file__)).parent))

import backend

print("import backend: {}".format(backend))
