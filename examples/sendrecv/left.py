import onnx_graphsurgeon as gs
import numpy as np
import onnx

X = gs.Variable(name="X", dtype=np.float32, shape=(2,3))

Y = gs.Variable(name="Y", dtype=np.float32, shape=(3,3))


node = gs.Node(op="SendRecv", attrs={"source":0,"destination":2,"shape":[3,3]}, inputs=[X], outputs=[Y],name = "send")

graph = gs.Graph(nodes=[node], inputs=[X], outputs=[Y])
onnx.save(gs.export_onnx(graph), "leftsendrecv.onnx")
