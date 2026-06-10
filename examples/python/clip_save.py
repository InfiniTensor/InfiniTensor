import sys
import onnx
import torch
import numpy as np
from pyinfinitensor.onnx import OnnxStub, backend
import torchvision.models as models

if __name__ == '__main__':
    model_path = 'model.onnx'
    print(model_path)

    onnx_model = onnx.load(model_path)
    print(onnx_model.graph.input)
    for input in onnx_model.graph.input:
        # 修改input_ids：固定为[1, 77]
        if input.name == "input_ids":
            input.type.tensor_type.shape.dim.clear()  # 清除原有动态维度
            # 添加固定维度（elem_type=7表示int64，无需修改）
            input.type.tensor_type.shape.dim.add().dim_value = 1    # text_batch_size=1
            input.type.tensor_type.shape.dim.add().dim_value = 77   # sequence_length=77
        
        # 修改pixel_values：固定为[1, 3, 224, 224]
        elif input.name == "pixel_values":
            input.type.tensor_type.shape.dim.clear()
            # 添加固定维度（elem_type=1表示float32，无需修改）
            input.type.tensor_type.shape.dim.add().dim_value = 1    # image_batch_size=1
            input.type.tensor_type.shape.dim.add().dim_value = 3    # num_channels=3
            input.type.tensor_type.shape.dim.add().dim_value = 224  # height=224
            input.type.tensor_type.shape.dim.add().dim_value = 224  # width=224
        
        # 修改attention_mask：固定为[1, 77]
        elif input.name == "attention_mask":
            input.type.tensor_type.shape.dim.clear()
            # 添加固定维度（与input_ids一致）
            input.type.tensor_type.shape.dim.add().dim_value = 1    # text_batch_size=1
            input.type.tensor_type.shape.dim.add().dim_value = 77   # sequence_length=77
    onnx.save(onnx_model, "multi_modal_model_fixed.onnx")  # 保存修改后的模型
    print("模型输入维度已修改为固定值并保存！")
    
