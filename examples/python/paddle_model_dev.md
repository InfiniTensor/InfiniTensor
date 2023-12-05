## Description

This is a doc to tell you how to run paddle*.py in your machine. If your model run on other machines except Nvidia, you may need to make some change.

## What do we do in paddle*.py files?

1. Train model and evalute model with Cifar10 dataset

2. Export paddle model to onnx model

3. Load onnx model, infer with InfiniTensor and calculate the inference accuracy

## Command

1. Go to `/examples/python` folder 

2. Run the following command
   
   1. ```
      python paddle_resnet.py
      python paddle_densenet.py
      python paddle_inception.py
      ```

## What should I do if I use other device(MLU, XPU, NPU)?

You need to change this code:

```
paddle.device.set_device("gpu") # Change gpu to mlu, xpu or npu
```
