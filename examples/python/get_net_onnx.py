import onnx
import torch
from onnxsim import simplify
import torchvision.models as models
import os
def export_and_analyze_model(pytorch_model, dummy_input, onnx_path, opset_version=11):
    """
    导出PyTorch模型到ONNX（固定batch size=1）并分析模型中的算子类型
    
    参数:
        pytorch_model: PyTorch模型实例
        dummy_input: 虚拟输入张量（batch size必须为1）
        onnx_path: 导出的ONNX文件路径
        opset_version: ONNX算子集版本，默认为11
    """
    # 确保模型处于评估模式
    pytorch_model.eval()
    
    # 检查输入batch size是否为1
    if dummy_input.shape[0] != 1:
        print(f"警告: 输入batch size为{dummy_input.shape[0]}，将重新生成batch size=1的输入")
        dummy_input = torch.randn(1, *dummy_input.shape[1:])
    
    print("=" * 60)
    print("开始导出PyTorch模型到ONNX格式（固定batch size=1）")
    print("=" * 60)
    
    # 导出ONNX模型 - 关键修改：不使用dynamic_axes
    torch.onnx.export(
        model=pytorch_model,
        args=dummy_input,
        f=onnx_path,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        # 移除了dynamic_axes参数，这样所有维度都是固定的
        # dynamic_axes参数被移除，batch size将固定为1
    )
    
    print(f"模型已成功导出到: {onnx_path}")
    print("Batch size已固定为1")
    print("-" * 40)
    
    # 可选：简化ONNX模型
    try:
        onnx_model = onnx.load(onnx_path)
        simplified_model, check = simplify(onnx_model)
        if check:
            simplified_path = onnx_path.replace('.onnx', '_simplified.onnx')
            onnx.save(simplified_model, simplified_path)
            print(f"简化模型已保存到: {simplified_path}")
            onnx_path = simplified_path  # 使用简化后的模型进行分析
    except Exception as e:
        print(f"模型简化过程中出现警告或错误: {e}")
        print("将继续使用原始模型进行分析")
    
    print("-" * 40)
    
    # 分析ONNX模型中的算子类型
    return analyze_onnx_operators(onnx_path)

def analyze_onnx_operators(onnx_path):
    """
    分析ONNX模型中的算子类型
    
    参数:
        onnx_path (str): ONNX模型文件的路径
        
    返回:
        set: 包含模型中所有算子类型的集合（自动去重）
    """
    # 加载ONNX模型
    model = onnx.load(onnx_path)
    
    # 获取计算图中的所有节点
    nodes = model.graph.node
    
    # 使用集合存储算子类型（自动去重）
    operator_types = set()
    
    # 遍历节点，提取op_type
    for node in nodes:
        operator_types.add(node.op_type)
    
    # 打印算子类型信息
    print(f"模型路径: {onnx_path}")
    print(f"算子类型数量: {len(operator_types)}")
    print("具体算子类型如下:")
    for op_type in sorted(operator_types):  # 按字母排序，便于阅读
        print(f"- {op_type}")
    
    return operator_types

def verify_fixed_batch_size(onnx_path):
    """
    验证ONNX模型的batch size是否已固定为1
    
    参数:
        onnx_path (str): ONNX模型文件的路径
    """
    model = onnx.load(onnx_path)
    
    print("验证batch size固定性:")
    for i, input in enumerate(model.graph.input):
        print(f"输入 {i} ({input.name}):")
        for j, dim in enumerate(input.type.tensor_type.shape.dim):
            if dim.dim_value == 0 and dim.dim_param:
                print(f"  维度 {j}: 动态 ({dim.dim_param})")
            else:
                print(f"  维度 {j}: 固定 ({dim.dim_value})")
    
    for i, output in enumerate(model.graph.output):
        print(f"输出 {i} ({output.name}):")
        for j, dim in enumerate(output.type.tensor_type.shape.dim):
            if dim.dim_value == 0 and dim.dim_param:
                print(f"  维度 {j}: 动态 ({dim.dim_param})")
            else:
                print(f"  维度 {j}: 固定 ({dim.dim_value})")

# 使用示例
if __name__ == "__main__":
    # 1. 加载预训练模型
    print("加载PyTorch模型...")
    # 加载预训练模型以Unet为例
    net = torch.hub.load('milesial/Pytorch-UNet', 'unet_carvana', pretrained=True, scale=0.5)
    net.eval()
    
    # 2. 准备虚拟输入（确保batch size=1）
    dummy_input = torch.randn(1, 3, 512, 512)  # batch size固定为1
    print(f"输入形状: {dummy_input.shape}")
    
    # 3. 导出并分析模型
    onnx_path = 'onnx_zoo/unet_carvana_pretrained_fixed_bs1.onnx'
    if not os.path.exists(onnx_path):
        os.makedirs(os.path.dirname(onnx_path), exist_ok=True)
   
    operator_types = export_and_analyze_model(
        pytorch_model=net,
        dummy_input=dummy_input,
        onnx_path=onnx_path,
        opset_version=11,
    )
    
    # 4. 验证batch size是否固定
    verify_fixed_batch_size(onnx_path)
    
    print("=" * 60)
    print("导出和分析完成！Batch size已固定为1")
    print("=" * 60)