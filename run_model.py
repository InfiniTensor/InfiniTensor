import sys
import onnx
import numpy as np
from pyinfinitensor.onnx import OnnxStub, backend
from onnxruntime import InferenceSession
import time
import nvtx

def initInputs(session: InferenceSession):
    """
    根据 ONNXRuntime 的实际输入生成随机输入数据
    """
    inputs = []
    for input_info in session.get_inputs():
        shape = [dim if dim > 0 else 1 for dim in input_info.shape]
        data_type = input_info.type  # 字符串形式，比如 'tensor(float)'
        np_type = toNumpyTypeFromString(data_type)
        inputs.append(np.random.random(shape).astype(np_type))
    return inputs

def toNumpyTypeFromString(type_str: str):
    """
    根据 ONNXRuntime 类型字符串转换为 numpy dtype
    """
    if "float" in type_str:
        return np.float32
    elif "double" in type_str:
        return np.float64
    elif "int32" in type_str:
        return np.int32
    elif "int64" in type_str:
        return np.int64
    elif "bool" in type_str:
        return np.bool_
    else:
        raise RuntimeError(f"Unsupported data type: {type_str}")

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python onnx_inference.py model_name.onnx")
        exit()
    model_path = sys.argv[1]

    # load model in ONNXRuntime
    session = InferenceSession(model_path)
    input_data = initInputs(session)

    # run on InfiniTensor
    onnx_model = onnx.load(model_path)
    # onnx_model = onnx.load("/home/zhangyunze/workspace/c610/fused_conv_sigmoid_mul.onnx")
    onnx_model = onnx.load("/home/zhangyunze/workspace/c610/yolov5s_fused_replace_conv_sigmoid_mul.onnx")
    model = OnnxStub(onnx_model, backend.cuda_runtime())
    for idata, itensor in zip(input_data, model.inputs.values()):
        itensor.copyin_numpy(idata)
    model.run()
    outputs = [output.copyout_numpy() for output in model.outputs.values()]
    print("InfiniTensor outputs:", len(outputs))
    print("Number of inputs:", len(input_data))

    # run ONNXRuntime
    ort_inputs = {input_info.name: idata for input_info, idata in zip(session.get_inputs(), input_data)}
    ort_outputs = session.run(None, ort_inputs)

    # compare results
    try:
        for i in range(len(outputs)):
            np.testing.assert_allclose(
                ort_outputs[i].flatten(), outputs[i].flatten(), rtol=1.0, atol=1e-05
            )
        print("The difference of results between ONNXRuntime and InfiniTensor looks good!")
    except AssertionError as e:
        print("Onnx and Infinitensor compare gap:" + str(e))

    # rng = nvtx.start_range("main_execution", color="green")
    infini_time = []
    for i in range(50):
        model.run()
    for i in range(1000):
        start_time = time.time()
        model.run()
        infini_time.append(time.time() - start_time)
    print(f"InfiniTensor推理时间: {np.mean(infini_time)*1000:.2f}ms")
    # # print(f"InfiniTensor推理时间优化后: {np.mean(infini_time)*1000:.2f}ms")
    # with nvtx.annotate("Complete Inference", color="red"):
        # model.run()
    # nvtx.end_range(rng)
    # onnx_model_1 = onnx.load(model_path)
    # model_1 = OnnxStub(onnx_model_1, backend.cuda_runtime())
    # for idata, itensor in zip(input_data, model_1.inputs.values()):
    #     itensor.copyin_numpy(idata)
    # for i in range(20):
    #     model_1.run()
    # infini_time_1 = []
    # for i in range(1000):
    #     start_time = time.time()
    #     model_1.run()
    #     infini_time_1.append(time.time() - start_time)
    # print(f"InfiniTensor推理时间优化前: {np.mean(infini_time_1)*1000:.2f}ms")
