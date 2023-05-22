# 以下分别导入精度检测包、数据DUMP包、AddPytorch算子测试用例生成与解析类，性能度量包
from InfiniTest.accuracy import Accuracy
from InfiniTest.dump import Dump
from InfiniTest.pytorch_operator import ConvPytorch
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
parser.add_argument("--generate_input_shape", default="1,3,224,224", type=str)
parser.add_argument("--generate_filter_shape", default="64,3,3,3", type=str)
parser.add_argument("--generate_param", default="1,1,1,1,1,1", type=str)
parser.add_argument("--parse", default="False", type=str, choices=["False", "True"])
args = parser.parse_args()

def generate_case(dir_path:str = "./conv/", input_shape:tuple = (1,3,224,224), filter_shape:tuple = (64,3,3,3), param:tuple = (1,1,1,1,1,1), type = torch.float32):
    # 构造tensor
    input = torch.rand(input_shape, dtype=type)
    filter = torch.rand(filter_shape, dtype=type)
    output = torch.nn.functional.conv2d(input, filter, padding = param[0:2], stride = param[2:4], dilation = param[4:6])
    # 构造测例实例
    conv = ConvPytorch([input,filter],[output],list(param[0:2]), list(param[2:4]), list(param[4:6]))
    # 序列化为文件
    file_path = dir_path + str(datetime.now()).replace(" ","_").replace("-","_").replace(":","_").replace(".","_") + ".prototxt" 
    conv.saveToFile(file_path, device= operator_pb2.DEVICE_GPU)
    logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)
    logging.info("\033[32m" + "[INFO] " + "\033[0m" + ": Save test case to path " + file_path)

def parse_case(dir_path:str="./conv/"):

    for file in os.listdir(dir_path):
        if os.path.isfile(dir_path + file) and os.path.splitext(dir_path + file)[1] =='.prototxt':
            check = ConvPytorch()
            inputs, outputs, pads, strides, dilations = check.loadFromFile(dir_path + file)
            handler = backend.GraphHandler(backend.cuda_runtime())
            input = handler.tensor(inputs[0].shape, 1)
            filter = handler.tensor(inputs[1].shape, 1)
            output = handler.conv(input,filter,None,pads[0],pads[1],strides[0],strides[1],dilations[0],dilations[1])
            handler.data_malloc()
            input.copyin_float(inputs[0].reshape(-1).tolist())
            filter.copyin_float(inputs[1].reshape(-1).tolist())
            handler.run()
            output = output.copyout_float()
            output = torch.tensor(output)
            output = torch.reshape(output,outputs[0].shape)
            acc = Accuracy()
            acc.computeDifference1(dir_path + file, output.numpy(), outputs[0].numpy())


if __name__ == "__main__":
    input_shape = args.generate_input_shape.split(",")
    input_shape = [int(i) for i in input_shape]
    input_shape = tuple(input_shape)
    filter_shape = args.generate_filter_shape.split(",")
    filter_shape = [int(i) for i in filter_shape]
    filter_shape = tuple(filter_shape)

    param = args.generate_param.split(",")
    param = [int(i) for i in param]
    param = tuple(param)
    for i in range(args.generate_num): 
        generate_case(input_shape = input_shape, filter_shape = filter_shape, param = param)
    if args.parse == "True":
        parse_case()

