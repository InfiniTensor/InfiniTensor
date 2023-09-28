import sys
import onnx
import torch
import numpy as np
from pyinfinitensor.onnx import OnnxStub, backend
import torchvision.models as models

if __name__ == '__main__':
    model_path = './resnet18.onnx'
    tv_model = models.resnet50(weights=None)
    input_shape = (1, 3, 224, 224)
    param = torch.rand(input_shape)
    torch.onnx.export(tv_model, param, model_path, verbose=False)

    onnx_model = onnx.load(model_path)
    model = OnnxStub(onnx_model, backend.cuda_runtime())
    images = np.random.random(input_shape).astype(np.float32)
    next(iter(model.inputs.values())).copyin_numpy(images)
    model.run()
    outputs = next(iter(model.outputs.values())).copyout_numpy()
    outputs = torch.tensor(outputs)
    outputs = torch.reshape(outputs, (1, 1000))
    _, predicted = torch.max(outputs, 1)
    print(predicted)
