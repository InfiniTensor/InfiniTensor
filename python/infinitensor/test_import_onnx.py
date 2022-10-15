from pyinfinitensor import *
from import_onnx import *

import sys

def main(netPath):
    runtime = CpuRuntimeObj.getInstance()
    graphFactory = GraphFactoryObj(runtime)
    import_onnx(graphFactory, netPath)

if __name__ == "__main__":
    main(sys.argv[1])