# 以下分别导入精度检测包、数据DUMP包、AddPytorch算子测试用例生成与解析类，性能度量包
from InfiniTest.accuracy import Accuracy
from InfiniTest.dump import Dump
from InfiniTest.pytorch_operator import SigmoidPytorch,ReluPytorch,SinPytorch,CosPytorch,TanPytorch,ASinPytorch,ACosPytorch,ATanPytorch,SinHPytorch,CosHPytorch,TanHPytorch,ASinHPytorch,ACosHPytorch,ATanHPytorch
from InfiniTest.performance import Profiling
from InfiniTest import operator_pb2
# 导入 pytoch、numpy、datatime、os、argparse、logging
import torch
import numpy
from datetime import datetime
import os
import argparse
import logging
# 导入本项目
from pyinfinitensor.onnx import backend

parser = argparse.ArgumentParser()
parser.add_argument("--generate_num", default="0", type=int)
parser.add_argument("--generate_operator", default="sigmoid", type=str, choices=["sigmoid", "relu"])
parser.add_argument("--generate_shape", default="1,3,2,3", type=str)
parser.add_argument("--parse", default="False", type=str, choices=["False", "True"])
args = parser.parse_args()

def generate_case(dir_path:str = "./unary/", operator:str = "sigmoid", shape:tuple = (1,3,2,3), type = torch.float32):
    # 构造tensor
    x = torch.rand(shape, dtype=type)
    if operator == "sigmoid":
        op = torch.nn.Sigmoid()
        z = op(x)
        opPytorch = SigmoidPytorch([x],[z])
    elif operator == "relu":
        op = torch.nn.ReLU()
        z = op(x)
        opPytorch = ReluPytorch([x],[z])
    # elif operator == "sin":
    #     op = torch.sin
    #     z = op(x)
    #     opPytorch = SinPytorch([x],[z])
    # elif operator == "cos":
    #     op = torch.cos
    #     z = op(x)
    #     opPytorch = CosPytorch([x],[z])
    # elif operator == "tan":
    #     op = torch.tan
    #     z = op(x)
    #     opPytorch = TanPytorch([x],[z])
    # elif operator == "asin":
    #     op = torch.asin
    #     z = op(x)
    #     opPytorch = ASinPytorch([x],[z])
    # elif operator == "acos":
    #     op = torch.acos
    #     z = op(x)
    #     opPytorch = ACosPytorch([x],[z])
    # elif operator == "atan":
    #     op = torch.atan
    #     z = op(x)
    #     opPytorch = ATanPytorch([x],[z])
    # elif operator == "sinh":
    #     op = torch.sinh
    #     z = op(x)
    #     opPytorch = SinHPytorch([x],[z])
    # elif operator == "cosh":
    #     op = torch.cosh
    #     z = op(x)
    #     opPytorch = CosHPytorch([x],[z])
    # elif operator == "tanh":
    #     op = torch.tanh
    #     z = op(x)
    #     opPytorch = TanHPytorch([x],[z])
    # elif operator == "asinh":
    #     op = torch.asinh
    #     z = op(x)
    #     opPytorch = ASinHPytorch([x],[z])
    # elif operator == "acosh":
    #     op = torch.acosh
    #     z = op(x)
    #     opPytorch = ACosHPytorch([x],[z])
    # elif operator == "atanh":
    #     op = torch.atanh
    #     z = op(x)
    #     opPytorch = ATanHPytorch([x],[z])

    # 序列化为文件
    file_path = dir_path + operator + "_" + str(datetime.now()).replace(" ","_").replace("-","_").replace(":","_").replace(".","_") + ".prototxt" 
    opPytorch.saveToFile(file_path, device= operator_pb2.DEVICE_GPU)
    logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)
    logging.info("\033[32m" + "[INFO] " + "\033[0m" + ": Save test case to path " + file_path)

def parse_case(dir_path:str="./unary/", operator:str = "sigmoid"):

    for file in os.listdir(dir_path):
        if os.path.isfile(dir_path + file) and os.path.splitext(dir_path + file)[1] =='.prototxt' and operator == file.split('_')[0]:
            if operator == "sigmoid":
                check = SigmoidPytorch()
                inputs, outputs = check.loadFromFile(dir_path + file)
                handler = backend.GraphHandler(backend.cuda_runtime())
                x = handler.tensor(inputs[0].shape, 1)
                z = handler.sigmoid(x,None)
            elif operator == "relu":
                check = ReluPytorch()
                inputs, outputs = check.loadFromFile(dir_path + file)
                handler = backend.GraphHandler(backend.cuda_runtime())
                x = handler.tensor(inputs[0].shape, 1)
                z = handler.relu(x,None)
            # elif operator == "sin":
            #     check = SinPytorch()
            #     inputs, outputs = check.loadFromFile(dir_path + file)
            #     handler = backend.GraphHandler(backend.cuda_runtime())
            #     x = handler.tensor(inputs[0].shape, 1)
            #     z = handler.sin(x,None)
            # elif operator == "cos":
            #     check = CosPytorch()
            #     inputs, outputs = check.loadFromFile(dir_path + file)
            #     handler = backend.GraphHandler(backend.cuda_runtime())
            #     x = handler.tensor(inputs[0].shape, 1)
            #     z = handler.cos(x,None)
            # elif operator == "tan":
            #     check = TanPytorch()
            #     inputs, outputs = check.loadFromFile(dir_path + file)
            #     handler = backend.GraphHandler(backend.cuda_runtime())
            #     x = handler.tensor(inputs[0].shape, 1)
            #     z = handler.tan(x,None)
            # elif operator == "asin":
            #     check = ASinPytorch()
            #     inputs, outputs = check.loadFromFile(dir_path + file)
            #     handler = backend.GraphHandler(backend.cuda_runtime())
            #     x = handler.tensor(inputs[0].shape, 1)
            #     z = handler.asin(x,None)
            # elif operator == "acos":
            #     check = ACosPytorch()
            #     inputs, outputs = check.loadFromFile(dir_path + file)
            #     handler = backend.GraphHandler(backend.cuda_runtime())
            #     x = handler.tensor(inputs[0].shape, 1)
            #     z = handler.acos(x,None)
            # elif operator == "atan":
            #     check = ATanPytorch()
            #     inputs, outputs = check.loadFromFile(dir_path + file)
            #     handler = backend.GraphHandler(backend.cuda_runtime())
            #     x = handler.tensor(inputs[0].shape, 1)
            #     z = handler.atan(x,None)
            # elif operator == "sinh":
            #     check = SinHPytorch()
            #     inputs, outputs = check.loadFromFile(dir_path + file)
            #     handler = backend.GraphHandler(backend.cuda_runtime())
            #     x = handler.tensor(inputs[0].shape, 1)
            #     z = handler.sinh(x,None)
            # elif operator == "cosh":
            #     check = CosHPytorch()
            #     inputs, outputs = check.loadFromFile(dir_path + file)
            #     handler = backend.GraphHandler(backend.cuda_runtime())
            #     x = handler.tensor(inputs[0].shape, 1)
            #     z = handler.cosh(x,None)
            # elif operator == "tanh":
            #     check = TanHPytorch()
            #     inputs, outputs = check.loadFromFile(dir_path + file)
            #     handler = backend.GraphHandler(backend.cuda_runtime())
            #     x = handler.tensor(inputs[0].shape, 1)
            #     z = handler.tanh(x,None)
            # elif operator == "asinh":
            #     check = ASinHPytorch()
            #     inputs, outputs = check.loadFromFile(dir_path + file)
            #     handler = backend.GraphHandler(backend.cuda_runtime())
            #     x = handler.tensor(inputs[0].shape, 1)
            #     z = handler.asinh(x,None)
            # elif operator == "acosh":
            #     check = ACosHPytorch()
            #     inputs, outputs = check.loadFromFile(dir_path + file)
            #     handler = backend.GraphHandler(backend.cuda_runtime())
            #     x = handler.tensor(inputs[0].shape, 1)
            #     z = handler.acosh(x,None)
            # elif operator == "atanh":
            #     check = ATanHPytorch()
            #     inputs, outputs = check.loadFromFile(dir_path + file)
            #     handler = backend.GraphHandler(backend.cuda_runtime())
            #     x = handler.tensor(inputs[0].shape, 1)
            #     z = handler.atanh(x,None)

            handler.data_malloc()
            x.copyin_float(inputs[0].reshape(-1).tolist())
            handler.run()
            z_out = z.copyout_float()
            z_out = torch.tensor(z_out)
            z_out = torch.reshape(z_out,outputs[0].shape)
            acc = Accuracy()
            acc.computeDifference1(dir_path + file, z_out.numpy(), outputs[0].numpy())


if __name__ == "__main__":
    shape = args.generate_shape.split(",")
    shape = [int(i) for i in shape]
    shape = tuple(shape)
    for i in range(args.generate_num): 
        generate_case(shape = shape, operator = args.generate_operator)
    if args.parse == "True":
        parse_case(operator = args.generate_operator)

