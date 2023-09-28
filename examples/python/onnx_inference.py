import sys
import onnx
import torch
import numpy as np
from pyinfinitensor.onnx import OnnxStub, backend

if __name__ == '__main__':
    args = sys.argv
    if len(sys.argv) != 2:
        print("Usage: python onnx_inference.py model_name.onnx")
        exit()
    model_path = sys.argv[1]
    # print(model_path)

    onnx_model = onnx.load(model_path)
    onnx_input = onnx_model.graph.input[0]
    input_shape = [[d.dim_value for d in _input.type.tensor_type.shape.dim]
                   for _input in onnx_model.graph.input]
    # Assume that there is only one input tensor
    input_shape = input_shape[0]
    # print(input_shape)
    input_data = np.random.random(input_shape).astype(np.float32)

    model = OnnxStub(onnx_model, backend.cuda_runtime())
    next(iter(model.inputs.values())).copyin_numpy(input_data)
    model.run()
    outputs = next(iter(model.outputs.values())).copyout_numpy()
    outputs = torch.tensor(outputs)
    print(outputs.shape)
