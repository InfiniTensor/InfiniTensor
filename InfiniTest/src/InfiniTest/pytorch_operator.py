import base_operator
import operator_pb2
import torch
import numpy

def datatype_switch(type):
    if type == numpy.float32:
        return torch.float32
    elif type == numpy.int32:
        return torch.int32
    elif type == numpy.float16:
        return torch.float16

class AddPytorch(base_operator.AddBase):
    def __init__(self, inputs:list=[], outputs:list=[]):
        inputs_list = [data.numpy() for data in inputs]
        outputs_list = [data.numpy() for data in outputs]
        inputs_layout_list = [operator_pb2.LAYOUT_NCHW, operator_pb2.LAYOUT_NCHW]
        outputs_layout_list = [operator_pb2.LAYOUT_NCHW]
        super().__init__(inputs_list,outputs_list,inputs_layout_list,outputs_layout_list)

    def loadFromFile(self, path, binary_file = False):
        inputs_dimension, inputs_stride, inputs_datatype, outputs_dimension, outputs_stride, outputs_datatype = super().loadFromFile(path, binary_file)
        inputs = []
        for input, shape, stride, datatype in zip(self.inputs, inputs_dimension, inputs_stride, inputs_datatype):
            temp = torch.tensor(input, dtype = datatype_switch(datatype))
            temp = temp.reshape(tuple(shape))
            if len(stride) > 0:
                temp = torch.as_strided(temp, tuple(shape), tuple(stride))
            inputs.append(temp)
        outputs = []
        for output, shape, stride, datatype in zip(self.outputs, outputs_dimension, outputs_stride, outputs_datatype):
            temp = torch.tensor(output, dtype = datatype_switch(datatype))
            temp = temp.reshape(tuple(shape))
            if len(stride) > 0:
                temp = torch.as_strided(temp, tuple(shape), tuple(stride))
            outputs.append(temp)
        return inputs, outputs


if __name__ == "__main__":
    input1 = torch.randint(0,100,(3,4),dtype=torch.int32)
    input2 = torch.randint(0,100,(3,4),dtype=torch.int32)
    output = torch.add(input1, input2)
    add = AddPytorch([input1,input2],[output])
    add.saveToFile("./prototxt",hex_option=False)
    add.saveToFile("./pb", binary_file = True)
    check = AddPytorch()
    inputs, outputs = check.loadFromFile("./prototxt")
    origin_result = torch.add(inputs[0],inputs[1])
    print (origin_result)
    print (outputs[0])
    inputs, outputs = check.loadFromFile("./pb",binary_file = True)
    origin_result = torch.add(inputs[0],inputs[1])
    print (origin_result)
    print (outputs[0])

