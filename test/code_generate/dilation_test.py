from cpp_module import *
import sys

dilaConfig = [[1, 1, 1, 1, 1, 1],
              [2, 2, 2, 2, 2, 2],
              [2, 2, 2, 4, 4, 4],
              [4, 4, 4, 4, 4, 4]]

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: ./dilation_test config_id batch_size\n"
              "config_id is in {0, 1, 2, 3}\n"
              "Example: ./dilation_test 0 1")
        exit(-1)
    configId = eval(sys.argv[1])
    n = eval(sys.argv[2])
    dc = dilaConfig[configId]

    g = Graph()
    i0 = g.tensor([n, 512, 14, 14], "FLOAT")
    i1 = g.tensor([n, 512, 14, 14], "FLOAT")
    i2 = g.tensor([n, 512, 14, 14], "FLOAT")
    i3 = g.tensor([n, 512, 14, 14], "FLOAT")
    i4 = g.tensor([n, 512, 14, 14], "FLOAT")
    i5 = g.tensor([n, 512, 14, 14], "FLOAT")
    i6 = g.tensor([n, 512, 14, 14], "FLOAT")
    i7 = g.tensor([n, 256, 14, 14], "FLOAT")
    i8 = g.tensor([n, 256, 14, 14], "FLOAT")
    i9 = g.tensor([n, 128, 14, 14], "FLOAT")
    i10 = g.tensor([n, 128, 14, 14], "FLOAT")
    i11 = g.tensor([n, 64, 14, 14], "FLOAT")
    i12 = g.tensor([n, 64, 14, 14], "FLOAT")

    w1 = g.tensor([512, 512, 3, 3], "FLOAT")
    w3 = g.tensor([512, 512, 3, 3], "FLOAT")
    w5 = g.tensor([512, 512, 3, 3], "FLOAT")
    w7 = g.tensor([256, 512, 3, 3], "FLOAT")
    w9 = g.tensor([128, 256, 3, 3], "FLOAT")
    w11 = g.tensor([64, 128, 3, 3], "FLOAT")

    g.conv(i0, w1, i1, dc[0], 1, 1, dc[0], dc[0])
    g.relu(i1, i2)
    g.conv(i2, w3, i3, dc[1], 1, 1, dc[1], dc[1])
    g.relu(i3, i4)
    g.conv(i4, w5, i5, dc[2], 1, 1, dc[2], dc[2])
    g.relu(i5, i6)
    g.conv(i6, w7, i7, dc[3], 1, 1, dc[3], dc[3])
    g.relu(i7, i8)
    g.conv(i8, w9, i9, dc[4], 1, 1, dc[4], dc[4])
    g.relu(i9, i10)
    g.conv(i10, w11, i11, dc[5], 1, 1, dc[5], dc[5])
    g.relu(i11, i12)

    g.setInputs([i0])
    g.setOutputs([i12])

    graph = SubGraph(g.getOperators())
    bestGraph = SubGraph()
    searchEngine = SearchEngine()
    searchEngine.run(graph, bestGraph)
    codeEngine = CodeEngine()
    perfEngine = searchEngine.exportPerfEngine()
    codeEngine.importPerfEngine(perfEngine)
    codeEngine.genCode(bestGraph, "res.cu")
