import sys
import onnx
import torch
import numpy as np
from pyinfinitensor.onnx import OnnxStub, backend
import sys
if __name__ == '__main__':
    args = sys.argv
    # if len(sys.argv) != 2:
    #     print("Usage: python onnx_inference.py model_name.onnx")
    #     exit()
    # model_path = sys.argv[1]
    model_path = 'multi_modal_model_fixed.onnx'
    print(model_path)

    onnx_model = onnx.load(model_path)
    onnx_input = onnx_model.graph.input[0]

    input_shape = [[d.dim_value for d in _input.type.tensor_type.shape.dim]
                   for _input in onnx_model.graph.input]
    # Assume that there is only one input tensor
    # print(input_shape)
    # # pråint(input_shape)

    # input_data = np.random.random(input_shape).astype(np.float32)
    # print(input_data)

    model = OnnxStub(onnx_model, backend.cuda_runtime())
    # print("Input shape:", input_shape)
    # print("Input data shape:", input_data.shape)
    # print("Expected size:", np.prod(input_shape))
    # print("Actual size:", input_data.size)
    
    # next(iter(model.inputs.values())).copyin_numpy(input_data)
    # model.run()
    # outputs = next(iter(model.outputs.values())).copyout_numpy()
    # outputs = torch.tensor(outputs)
    # print(outputs.shape)
