import onnxruntime as ort
import numpy as np
import time
from onnx import ModelProto, ValueInfoProto


from onnxruntime import InferenceSession
# 模型路径
model_path = r"/home/yswang/winter_learn/ai_compiler/InfiniTensor/pyinfinitensor/tests/resnet50-v2-7.onnx"
model_path1=r"/home/yswang/winter_learn/ai_compiler/InfiniTensor/pyinfinitensor/tests/optimized.onnx"


# 加载模型
session = ort.InferenceSession(model_path)
from onnx import version_converter, load, save

try:
    model = load(model_path1)
    model_opset = model.opset_import[0].version
    if model_opset > 21:
        print(f"模型使用的 Opset 是 {model_opset}，正在降级为 21...")
        converted = version_converter.convert_version(model, 21)
        downgraded_path = model_path1.replace(".onnx", "_opset21.onnx")
        save(converted, downgraded_path)
        model_path1 = downgraded_path
except Exception as e:
    print(f"降级模型失败: {e}")
session1 = ort.InferenceSession(model_path1)


# 获取模型的输入信息
input_name = session.get_inputs()[0].name
input_name1 = session1.get_inputs()[0].name
input_shape = session.get_inputs()[0].shape
input_dtype = session.get_inputs()[0].type

# 将数据类型映射到 numpy
def to_numpy_dtype(onnx_dtype):
    return {
        'tensor(float)': np.float32,
        'tensor(double)': np.float64,
        'tensor(int32)': np.int32,
        'tensor(int64)': np.int64
    }.get(onnx_dtype, np.float32)

# 创建随机输入数据
input_data = np.random.randn(*[dim if isinstance(dim, int) else 1 for dim in input_shape]).astype(to_numpy_dtype(input_dtype))

# 运行并计时
start = time.time()
outputs = session.run(None, {input_name: input_data})
end = time.time()

print(f"Inference time: {end - start:.6f} seconds")

start = time.time()
outputs = session1.run(None, {input_name1: input_data})
end = time.time()
print(f"Inference1 time: {end - start:.6f} seconds")
