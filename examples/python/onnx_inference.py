import sys
import onnx
from pyinfinitensor.onnx import OnnxStub, backend
import torch

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
    input_data = torch.randint(0, 100, input_shape, dtype=torch.float32)

    model = OnnxStub(onnx_model, backend.cuda_runtime())
    next(model.inputs.items().__iter__())[
        1].copyin_float(input_data.reshape(-1).tolist())
    model.run()
    outputs = next(model.outputs.items().__iter__())[1].copyout_float()
    outputs = torch.tensor(outputs)
    print(outputs.shape)
