import os, onnx, unittest
from typing import  Dict
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from PIL import Image
from onnx import TensorProto
from onnx.helper import (
    make_model,
    make_node,
    make_tensor,
    make_graph,
    make_tensor_value_info,
)
from onnx.checker import check_model
from pyinfinitensor.onnx import from_onnx, backend, run_onnx
import onnxruntime

def pre_process(img) :
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    transfn = transforms.Compose([transforms.Resize(256), 
                              transforms.CenterCrop(224),
                              transforms.ToTensor(),
                              normalize,
                              ])
    #img = np.transpose(img,(1,2,0))
    return transfn(img)    

def model_run(onnx_file_name):
    dir_path = "./"
    files = os.listdir(dir_path)
    model_file = next(
        (name for name in files if name.endswith(onnx_file_name)), None
    )
    
    if model_file != None:
        model_path = os.path.join(dir_path, model_file)
        print(
            "model: {file}({size:.2f} MiB)".format(
                file=model_path, size=os.path.getsize(model_path) / 1024 / 1024
            )
        )
        model = onnx.load(model_path)
        check_model(model)        
        
        session = onnxruntime.InferenceSession(model.SerializeToString(), None)
        input_name = session.get_inputs()[0].name       
        input_tensor = make_tensor(input_name, TensorProto.FLOAT, input_data.shape, input_data)
        run_onnx(model,  [input_tensor])  

def get_img_data():
    dir_path = "./"
    files = os.listdir(dir_path)
    img_file = next(
        (name for name in files if name.endswith(".jpg")), None
        )
    if img_file != None:
        img_path = os.path.join(dir_path, img_file)
        try:
            img = Image.open(img_path)
            input_data = pre_process(img)
            img.close()
        except FileNotFoundError:
            print(f"Image not found: {img_file}")
        else: 
            return input_data.numpy().astype(np.float32)
class TestStringMethods(unittest.TestCase):    
    def test_model_run(self):
        #input_data = get_img_data()
        input_data = np.random.rand(1,3,224,224).astype(np.float32)
        model_run("resnet18.onnx", input_data)
        model_run("resnet50.onnx", input_data)
        model_run("drn_c_26.onnx", input_data)



                

if __name__ == "__main__":
    unittest.main()
