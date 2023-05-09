import operator_pb2
import struct
import numpy
from google.protobuf import text_format

def float_to_hex(f):
    return float.hex(f)
def int32_to_hex(i):
    return hex(i)
def hex_to_float(h):
    return float.fromhex(h)
def hex_to_int32(h):
    return int(h,16)

def datatype_switch_from_numpy(type):
    if type == numpy.float32:
        return operator_pb2.DTYPE_FLOAT
    elif type == numpy.int32:
        return operator_pb2.DTYPE_INT32
    elif type == numpy.float16:
        return operator_pb2.DTYPE_HALF
    else:
        return operator_pb2.DTYPE_UNKNOWN

def datatype_switch_from_operator(type):
    if type == operator_pb2.DTYPE_FLOAT:
        return numpy.float32
    elif type == operator_pb2.DTYPE_INT32:
        return numpy.int32
    elif type == operator_pb2.DTYPE_HALF:
        return numpy.float16

class Operator(object):
    def __init__(self, inputs:list=[], outputs:list=[], inputs_layout:list=[], outputs_layout:list=[]):
        self.inputs = inputs
        self.outputs = outputs
        self.inputs_layout = inputs_layout
        self.outputs_layout = outputs_layout

    def loadFromFile(self, path, binary_file = False):
        operator = operator_pb2.Operator()
        if binary_file == True:
            with open(path, "rb") as f:
                operator.ParseFromString(f.read())
            f.close()
        else:
            with open(path, "r") as f:
                operator = text_format.Parse(f.read(), operator_pb2.Operator())
            f.close()
        self.name = operator.name
        self.device = operator.device
        
        self.inputs.clear()
        self.inputs_layout.clear()

        inputs_dimension = []
        inputs_datatype = []
        inputs_stride = []
        outputs_dimension = []
        outputs_stride = []
        outputs_datatype = []

        for input in operator.inputs:
            if len(input.valueFloat) > 0:
                self.inputs.append(input.valueFloat)
            elif len(input.valueHex) > 0:
                if input.datatype == operator_pb2.DTYPE_FLOAT:
                    self.inputs.append([hex_to_float(data) for data in input.valueHex])
                elif input.datatype == operator_pb2.DTYPE_INT32:
                    self.inputs.append([hex_to_int32(data) for data in input.valueHex])
            elif len(input.valueInt32) > 0:
                self.inputs.append(input.valueInt32)
            self.inputs_layout.append(input.layout)
            inputs_dimension.append(input.shape.dimension)
            inputs_stride.append(input.shape.stride)
            inputs_datatype.append(datatype_switch_from_operator(input.datatype))

        self.outputs.clear()
        self.outputs_layout.clear()
        for output in operator.outputs:
            if len(output.valueFloat) > 0:
                self.outputs.append(output.valueFloat)
            elif len(output.valueHex) > 0:
                if output.datatype == operator_pb2.DTYPE_FLOAT:
                    self.outputs.append([hex_to_float(data) for data in output.valueHex])
                elif output.datatype == operator_pb2.DTYPE_INT32:
                    self.outputs.append([hex_to_int32(data) for data in output.valueHex])
            elif len(output.valueInt32) > 0:
                self.outputs.append(output.valueInt32)
            self.outputs_layout.append(output.layout)
            outputs_dimension.append(output.shape.dimension)
            outputs_stride.append(output.shape.stride)
            outputs_datatype.append(datatype_switch_from_operator(output.datatype))

        return inputs_dimension, inputs_stride, inputs_datatype, outputs_dimension, outputs_stride, outputs_datatype

    def saveToFile(self, path, hex_option:bool = False, binary_file = False):
        operator = operator_pb2.Operator()
        operator.name = self.name
        operator.device = operator_pb2.DEVICE_CPU

        for input, layout in zip(self.inputs, self.inputs_layout):
            input_tensor = operator_pb2.Tensor()
            input_tensor.layout = layout
            input_tensor.datatype = datatype_switch_from_numpy(input.dtype)
            input_tensor.shape.dimension.extend(list(input.shape))
            if hex_option == True:
                input_list = input.reshape(-1).tolist()
                if input.dtype == numpy.float32:
                    input_hex = [float_to_hex(data) for data in input_list] 
                elif input.dtype == numpy.int32:
                    input_hex = [int32_to_hex(data) for data in input_list]
                input_tensor.valueHex.extend(input_hex) 
            else:
                input_list = input.reshape(-1).tolist()
                if input.dtype == numpy.float32:
                    input_tensor.valueFloat.extend(input_list) 
                elif input.dtype == numpy.int32:
                    input_tensor.valueInt32.extend(input_list) 
            operator.inputs.append(input_tensor)

        for output, layout in zip(self.outputs, self.outputs_layout):
            output_tensor = operator_pb2.Tensor()
            output_tensor.layout = layout
            output_tensor.datatype = datatype_switch_from_numpy(output.dtype)
            output_tensor.shape.dimension.extend(list(output.shape))
            if hex_option == True:
                output_list = output.reshape(-1).tolist()
                if output.dtype == numpy.float32:
                    output_hex = [float_to_hex(data) for data in output_list] 
                elif output.dtype == numpy.int32:
                    output_hex = [int32_to_hex(data) for data in output_list] 
                output_tensor.valueHex.extend(output_hex) 
            else:
                output_list = output.reshape(-1).tolist()
                if output.dtype == numpy.float32:
                    output_tensor.valueFloat.extend(output_list) 
                elif output.dtype == numpy.int32:
                    output_tensor.valueInt32.extend(output_list) 
            operator.outputs.append(output_tensor)

        if binary_file == True:
            with open(path, "wb") as f:
                f.write(operator.SerializeToString())
            f.close()
        else:
            with open(path, "w") as f:
                f.write(str(operator))
            f.close()

class AddBase(Operator):
    def __init__(self, inputs:list=[], outputs:list=[], inputs_layout:list=[], outputs_layout:list=[]):
        super().__init__(inputs, outputs, inputs_layout, outputs_layout)
        self.name = "Add"

    def loadFromFile(self, path, binary_file = False):
        inputs_dimension, inputs_stride, inputs_datatype, outputs_dimension, outputs_stride, outputs_datatype = super().loadFromFile(path, binary_file)
        return inputs_dimension, inputs_stride, inputs_datatype, outputs_dimension, outputs_stride, outputs_datatype

if __name__ == "__main__":
    input1 = numpy.array([1.0,2.0,3.0], dtype=numpy.float32)
    input2 = numpy.array([1.0,2.0,3.0], dtype=numpy.float32)
    inputs = [input1, input2]
    inputs_layout = [operator_pb2.LAYOUT_NCHW, operator_pb2.LAYOUT_NCHW]
    output = numpy.array([a+b for a,b in zip(input1, input2)])
    outputs = [output]
    outputs_layout = [operator_pb2.LAYOUT_NCHW]
    a = AddBase(inputs, outputs, inputs_layout, outputs_layout);
    a.saveToFile("./prototxt",hex_option=True)
    a.saveToFile("./pb", binary_file = True)
    check = AddBase()
    check.loadFromFile("./pb",binary_file = True)
    check.loadFromFile("./prototxt")
