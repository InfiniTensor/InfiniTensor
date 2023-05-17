# 以下分别导入精度检测包、数据DUMP包、AddPytorch算子测试用例生成与解析类，性能度量包
from InfiniTest.accuracy import Accuracy
from InfiniTest.dump import Dump
from InfiniTest.pytorch_operator import MatmulPytorch
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
parser.add_argument("--generate_left_shape", default="2,3", type=str)
parser.add_argument("--generate_right_shape", default="3,4", type=str)
parser.add_argument("--parse", default="False", type=str, choices=["False", "True"])
args = parser.parse_args()

def generate_case(dir_path:str = "./matmul/", left_shape:tuple = (2,3), right_shape:tuple = (3,4), type = torch.float32):
    # 构造tensor
    x = torch.rand(left_shape, dtype=type)
    y = torch.rand(right_shape, dtype=type)
    x_gpu = x.to(torch.device("cuda"))
    y_gpu = y.to(torch.device("cuda"))
    z_gpu = torch.matmul(x_gpu, y_gpu)
    z = z_gpu.to(torch.device("cpu"))
    # 构造测例实例
    matmul = MatmulPytorch([x,y],[z])
    # 序列化为文件
    file_path = dir_path + str(datetime.now()).replace(" ","_").replace("-","_").replace(":","_").replace(".","_") + ".prototxt" 
    matmul.saveToFile(file_path, device= operator_pb2.DEVICE_GPU)
    logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)
    logging.info("\033[32m" + "[INFO] " + "\033[0m" + ": Save test case to path " + file_path)

def parse_case(dir_path:str="./matmul/"):

    for file in os.listdir(dir_path):
        if os.path.isfile(dir_path + file) and os.path.splitext(dir_path + file)[1] =='.prototxt':
            check = MatmulPytorch()
            inputs, outputs = check.loadFromFile(dir_path + file)
            handler = backend.GraphHandler(backend.cuda_runtime())
            x = handler.tensor(inputs[0].shape, 1)
            y = handler.tensor(inputs[1].shape, 1)
            z = handler.matmul(x,y,None,False,False,None,backend.ActType.Linear)
            handler.data_malloc()
            x.copyin_float(inputs[0].reshape(-1).tolist())
            y.copyin_float(inputs[1].reshape(-1).tolist())
            handler.run()
            z_out = z.copyout_float()
            z_out = torch.tensor(z_out)
            z_out = torch.reshape(z_out,outputs[0].shape)
            acc = Accuracy()
            acc.computeDifference1(dir_path + file, z_out.numpy(), outputs[0].numpy())


if __name__ == "__main__":
    left_shape = args.generate_left_shape.split(",")
    left_shape = [int(i) for i in left_shape]
    left_shape = tuple(left_shape)
    right_shape = args.generate_right_shape.split(",")
    right_shape = [int(i) for i in right_shape]
    right_shape = tuple(right_shape)
    for i in range(args.generate_num): 
        generate_case(left_shape = left_shape, right_shape = right_shape)
    if args.parse == "True":
        parse_case()

