import sys
import onnx
import torch
import numpy as np
from pyinfinitensor.onnx import OnnxStub, backend
import time

if __name__ == '__main__':
    model_path = 'onnx_zoo/unet_carvana_pretrained_fixed_bs1_simplified.onnx'
    print(model_path)

    onnx_model = onnx.load(model_path)
    onnx_input = onnx_model.graph.input[0]

    # 获取输入形状 - 修复部分
    input_shape = [[d.dim_value for d in _input.type.tensor_type.shape.dim]
                   for _input in onnx_model.graph.input]
    
    # 选择第一个输入的形状并确保是整数列表
    input_shape = input_shape[0]  # 取第一个输入的形状
    print("Input shape:", input_shape)
    
    # 固定随机种子，确保两次生成的输入数据一致
    np.random.seed(42)
    
    # 修复：使用 *input_shape 解包形状参数
    input_data_numpy = np.random.random(tuple(input_shape)).astype(np.float32)
    dummy_input_torch = torch.from_numpy(input_data_numpy)
    input_data = input_data_numpy

    # 加载PyTorch模型（如果使用onnx2torch）
    try:
        import onnx2torch
        net = onnx2torch.convert(onnx_model)
        net.eval()
        
        # PyTorch 推理
        with torch.no_grad():
            output_torch = net(dummy_input_torch)
        print(f"PyTorch output shape: {output_torch.shape}")
    except ImportError:
        print("onnx2torch not installed, skipping PyTorch comparison")
        # 如果没有onnx2torch，跳过PyTorch部分
        pass
    except Exception as e:
        print(f"Error converting ONNX to PyTorch: {e}")
        # 如果转换失败，跳过PyTorch部分
        pass

    # 您的框架推理
    model = OnnxStub(onnx_model, backend.cuda_runtime())
    next(iter(model.inputs.values())).copyin_numpy(input_data)
    model.run()
    outputs = next(iter(model.outputs.values())).copyout_numpy()
    outputs = torch.tensor(outputs)
    print(f"Your Framework output shape: {outputs.shape}")

    # 如果成功运行了PyTorch部分，则进行对比
    try:
        # 结果对比
        if 'output_torch' in locals():
            if output_torch.shape == outputs.shape:
                print("✅ Output shapes match.")
            else:
                print(f"❌ Output shapes do not match! PyTorch: {output_torch.shape}, Yours: {outputs.shape}")
                
            # 数值对比
            absolute_tolerance = 1e-5
            relative_tolerance = 1e-5

            if torch.allclose(output_torch, outputs, atol=absolute_tolerance, rtol=relative_tolerance):
                print("✅ Output values match within tolerance.(In 1e-5 tolerance)")
            else:
                print("❌ Output values differ beyond the specified tolerance.")
                abs_diff = torch.abs(output_torch - outputs)
                max_abs_diff = torch.max(abs_diff).item()
                mean_abs_diff = torch.mean(abs_diff).item()
                print(f"Absolute difference: {abs_diff}")
                print(f"   Max absolute difference: {max_abs_diff:.7f}")
                print(f"   Mean absolute difference: {mean_abs_diff:.7f}")
    except:
        print("Skipping comparison as PyTorch model is not available")