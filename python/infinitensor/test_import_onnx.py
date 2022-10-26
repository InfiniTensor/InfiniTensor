from pyinfinitensor import *
from import_onnx import *

import sys

def main(netPath):
    runtime = CpuRuntimeObj.getInstance()
    graphBuilder = GraphBuilderObj(runtime)
    import_onnx(graphBuilder, netPath)

if __name__ == "__main__":
    main(sys.argv[1])
