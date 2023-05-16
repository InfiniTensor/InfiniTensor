# 以下分别导入精度检测包、数据DUMP包、AddPytorch算子测试用例生成与解析类，性能度量包
from InfiniTest.accuracy import Accuracy
from InfiniTest.dump import Dump
from InfiniTest.pytorch_operator import AddPytorch
from InfiniTest.performance import Profiling
from InfiniTest import operator_pb2
# 导入 pytoch、numpy、datatime、os、argparse
import torch
import numpy
from datetime import datetime
import os
import argparse
# 导入本项目
from pyinfinitensor.onnx import backend

parser = argparse.ArgumentParser()
parser.add_argument("--generate_num", default="0", type=int)
parser.add_argument("--parse", default="False", type=str, choices=["False", "True"])
args = parser.parse_args()

def generate_case(dir_path:str = "./add/", shape:tuple = (3,4), type = torch.float32):
    # 构造tensor
    x = torch.rand(shape, dtype=type)
    y = torch.rand(shape, dtype=type)
    x_gpu = x.to(torch.device("cuda"))
    y_gpu = y.to(torch.device("cuda"))
    z_gpu = torch.add(x_gpu, y_gpu)
    z = z_gpu.to(torch.device("cpu"))
    # 构造测例实例
    add = AddPytorch([x,y],[z])
    # 序列化为文件
    add.saveToFile(dir_path + str(datetime.now()).replace(" ","_").replace("-","_").replace(":","_").replace(".","_") + ".prototxt", device= operator_pb2.DEVICE_GPU)

def parse_case(dir_path:str="./add/"):

    for file in os.listdir(dir_path):
        if os.path.isfile(dir_path + file) and os.path.splitext(dir_path + file)[1] =='.prototxt':
            check = AddPytorch()
            inputs, outputs = check.loadFromFile(dir_path + file)
            handler = backend.GraphHandler(backend.cuda_runtime())
            x = handler.tensor(inputs[0].shape, 1)
            y = handler.tensor(inputs[1].shape, 1)
            z = handler.add(x,y,None)
            handler.data_malloc()
            x.copyin_float(inputs[0].reshape(-1).tolist())
            y.copyin_float(inputs[1].reshape(-1).tolist())
            handler.run()
            z_out = z.copyout_float()
            z_out = torch.tensor(z_out)
            z_out = torch.reshape(z_out,outputs[0].shape)
            acc = Accuracy()
            acc.computeDifference0(dir_path + file, z_out.numpy(), outputs[0].numpy())


if __name__ == "__main__":
    for i in range(args.generate_num): 
        generate_case()
    if args.parse == "True":
        parse_case()

