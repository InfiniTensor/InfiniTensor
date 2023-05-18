# 以下分别导入精度检测包、数据DUMP包、AddPytorch算子测试用例生成与解析类，性能度量包
from InfiniTest.accuracy import Accuracy
from InfiniTest.dump import Dump
from InfiniTest.pytorch_operator import SigmoidPytorch
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
parser.add_argument("--generate_shape", default="2,3", type=str)
parser.add_argument("--parse", default="False", type=str, choices=["False", "True"])
args = parser.parse_args()

def generate_case(dir_path:str = "./sigmoid/", shape:tuple = (3,4), type = torch.float32):
    # 构造tensor
    x = torch.rand(shape, dtype=type)
    sigmoid_operator = torch.nn.Sigmoid()
    z = sigmoid_operator(x)
    # 构造测例实例
    sigmoid = SigmoidPytorch([x],[z])
    # 序列化为文件
    file_path = dir_path + str(datetime.now()).replace(" ","_").replace("-","_").replace(":","_").replace(".","_") + ".prototxt" 
    sigmoid.saveToFile(file_path, device= operator_pb2.DEVICE_GPU)
    logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)
    logging.info("\033[32m" + "[INFO] " + "\033[0m" + ": Save test case to path " + file_path)

def parse_case(dir_path:str="./sigmoid/"):

    for file in os.listdir(dir_path):
        if os.path.isfile(dir_path + file) and os.path.splitext(dir_path + file)[1] =='.prototxt':
            check = SigmoidPytorch()
            inputs, outputs = check.loadFromFile(dir_path + file)
            handler = backend.GraphHandler(backend.bang_runtime())
            x = handler.tensor(inputs[0].shape, 1)
            z = handler.sigmoid(x,None)
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
        generate_case(shape = shape)
    if args.parse == "True":
        parse_case()

