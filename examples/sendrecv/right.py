import onnx_graphsurgeon as gs
import numpy as np
import onnx



W = gs.Variable(name="W", dtype=np.float32, shape=(3,3))
Y = gs.Variable(name="Y", dtype=np.float32, shape=(2,3))



node = gs.Node(op="SendRecv", attrs={"source":0,"destination":2,"shape":[2,3]}, inputs=[W], outputs=[Y],name = "recv")

graph = gs.Graph(nodes=[node], inputs=[W], outputs=[Y])
onnx.save(gs.export_onnx(graph), "rightsendrecv.onnx")
