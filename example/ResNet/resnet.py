# 引入 torch 包、神经网络包、函数包、优化器
import torch
import torch.nn as nn
import torch.nn.functional as func
import torch.optim as optim
# 引入图像包、图像处理包、显示包、时间包
import torchvision
import torchvision.transforms as transforms
from torchvision.transforms import ToPILImage
import time
from torch.utils.data import Sampler
import random
# 引入 onnx 相关工具
import onnx
from onnxsim import simplify
from onnx import version_converter
from onnx2torch import convert
# 引入参数解析包
import argparse
# 引入工具包
import sys
from tqdm import tqdm

# 定义参数解析
parser = argparse.ArgumentParser()
parser.add_argument("--train_batch", default="4", type=int, help="Default value of train_batch is 4.")
parser.add_argument("--train_epoch", default="2", type=int, help="Default value of train_epoch is 2. Set 0 to this argument if you want an untrained network.")
parser.add_argument("--infer_batch", default="1", type=int, help="Default value of infer_batch is 1.")
parser.add_argument("--which_net", type=str, required=True, choices=["resnet18", "resnet34", "resnet50", "resnet101", "resnet152"])
parser.add_argument("--which_device", type=str, default="cpu", choices=["mlu", "cuda", "cpu", "xpu"])
parser.add_argument("--pytorch_infer", type=str, default="False", choices=["False", "True"])
parser.add_argument("--gofusion_infer", type=str, default="False", choices=["False", "True"])
parser.add_argument("--magicmind_infer", type=str, default="False", choices=["False", "True"])
parser.add_argument("--onnx_file", type=str, default="")
parser.add_argument("--image_size", default="224", type=int, help="The width/height of image")
parser.add_argument("--sample", default="1", type=float, help="The percentage of the all images which to be infer.")
args = parser.parse_args()

print(vars(args))

# 网络简化工具
from typing import Any, List, Optional
from onnx import ModelProto, NodeProto

def eliminate_batchnorm(model: ModelProto):
    class LinkedTensor:
        node: NodeProto
        slot: int

        def __init__(self, node: NodeProto, slot: int) -> None:
            self.node = node
            self.slot = slot

    tensors: dict[str, LinkedTensor] = {}
    targets: dict[str, list[LinkedTensor]] = {}
    batchnorm: list[NodeProto] = []

    for node in model.graph.node:
        for j, tensor in enumerate(node.output):
            tensors[tensor] = LinkedTensor(node, j)
        if node.op_type == "BatchNormalization":
            batchnorm.append(node)
        else:
            for slot, t in enumerate(node.input):
                source = tensors.get(t)
                if source != None and source.node.op_type == "BatchNormalization":
                    targets.setdefault(source.node.name, []).append(
                        LinkedTensor(node, slot)
                    )

    for node in batchnorm:
        for target in targets[node.name]:
            target.node.input[target.slot] = node.input[0]
        model.graph.node.remove(node)


# 定义网络并显示
class BasicBlock(nn.Module):
    """Basic Block for resnet 18 and resnet 34
    """

    #BasicBlock and BottleNeck block
    #have different output size
    #we use class attribute expansion
    #to distinct
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        #residual function
        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels * BasicBlock.expansion, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels * BasicBlock.expansion)
        )

        #shortcut
        self.shortcut = nn.Sequential()

        #the shortcut output dimension is not the same with residual function
        #use 1*1 convolution to match the dimension
        if stride != 1 or in_channels != BasicBlock.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * BasicBlock.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * BasicBlock.expansion)
            )

    def forward(self, x):
        return nn.ReLU(inplace=True)(self.residual_function(x) + self.shortcut(x))

class BottleNeck(nn.Module):
    """Residual block for resnet over 50 layers
    """
    expansion = 4
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, stride=stride, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels * BottleNeck.expansion, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels * BottleNeck.expansion),
        )

        self.shortcut = nn.Sequential()

        if stride != 1 or in_channels != out_channels * BottleNeck.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * BottleNeck.expansion, stride=stride, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels * BottleNeck.expansion)
            )

    def forward(self, x):
        return nn.ReLU(inplace=True)(self.residual_function(x) + self.shortcut(x))

class ResNet(nn.Module):

    def __init__(self, block, num_block, num_classes=100):
        super().__init__()

        self.in_channels = 64

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True))
        #we use a different inputsize than the original paper
        #so conv2_x's stride is 1
        self.conv2_x = self._make_layer(block, 64, num_block[0], 1)
        self.conv3_x = self._make_layer(block, 128, num_block[1], 2)
        self.conv4_x = self._make_layer(block, 256, num_block[2], 2)
        self.conv5_x = self._make_layer(block, 512, num_block[3], 2)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, out_channels, num_blocks, stride):
        """make resnet layers(by layer i didnt mean this 'layer' was the
        same as a neuron netowork layer, ex. conv layer), one layer may
        contain more than one residual block
        Args:
            block: block type, basic block or bottle neck block
            out_channels: output depth channel number of this layer
            num_blocks: how many blocks per layer
            stride: the stride of the first block of this layer
        Return:
            return a resnet layer
        """

        # we have num_block blocks per layer, the first block
        # could be 1 or 2, other blocks would always be 1
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        output = self.conv1(x)
        output = self.conv2_x(output)
        output = self.conv3_x(output)
        output = self.conv4_x(output)
        output = self.conv5_x(output)
        output = self.avg_pool(output)
        output = output.view(output.size(0), -1)
        output = self.fc(output)

        return output

def resnet18():
    """ return a ResNet 18 object
    """
    return ResNet(BasicBlock, [2, 2, 2, 2], 10)

def resnet34():
    """ return a ResNet 34 object
    """
    return ResNet(BasicBlock, [3, 4, 6, 3], 10)

def resnet50():
    """ return a ResNet 50 object
    """
    return ResNet(BottleNeck, [3, 4, 6, 3], 10)

def resnet101():
    """ return a ResNet 101 object
    """
    return ResNet(BottleNeck, [3, 4, 23, 3], 10)

def resnet152():
    """ return a ResNet 152 object
    """
    return ResNet(BottleNeck, [3, 8, 36, 3], 10)

# 定义数据预处理方式以及训练集与测试集并进行下载
class PercentageSampler(Sampler):
    def __init__(self, data_source, percentage):
        self.data_source = data_source
        self.percentage = percentage
        self.num_samples = int(len(data_source) * percentage) // args.infer_batch * args.infer_batch

    def __iter__(self):
        indices = list(range(len(self.data_source)))
        random.shuffle(indices)
        return iter(indices[:self.num_samples])

    def __len__(self):
        return self.num_samples

transform = transforms.Compose([
        transforms.Resize((args.image_size,args.image_size)),
        transforms.ToTensor(), # 转为Tensor
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), # 归一化
                             ])
trainset = torchvision.datasets.CIFAR10(
                    root='../data', 
                    train=True, 
                    download=True,
                    transform=transform)
trainloader = torch.utils.data.DataLoader(
                    trainset, 
                    batch_size=args.train_batch,
                    shuffle=True, 
                    num_workers=2)
testset = torchvision.datasets.CIFAR10(
                    '../data',
                    train=False, 
                    download=True, 
                    transform=transform)
sampler = PercentageSampler(testset, args.sample)
testloader = torch.utils.data.DataLoader(
                    testset,
                    batch_size=args.infer_batch, 
                    shuffle=False,
                    num_workers=2,
                    sampler = sampler)
########################################################################
# 定义网络
if args.which_net == "resnet18":
    net = resnet18()
elif args.which_net == "resnet34":
    net = resnet34()
elif args.which_net == "resnet50":
    net = resnet50()
elif args.which_net == "resnet101":
    net = resnet101()
elif args.which_net == "resnet152":
    net = resnet152()
else:
    print("Non-existent network")
    sys.exit()

evalnet = net
evalnet.eval()
# 定义损失函数与优化器
criterion = nn.CrossEntropyLoss() # 交叉熵损失函数
optimizer = optim.SGD(evalnet.parameters(), lr=0.001, momentum=0.9)

# 定义硬件设备
input_rand = torch.zeros((args.infer_batch,3,args.image_size,args.image_size))
torch.onnx.export(evalnet, input_rand, args.which_net + '_untrained' + '.onnx', input_names = ["image"], output_names = ["label"])

# 网络训练
if args.train_epoch != 0:
    device = torch.device(args.which_device)
    net = net.to(device)
    if args.which_device == "mlu" or args.which_device == "xpu" :
        print("[INFO] Pytorch doesn't support network train on " + args.which_device)
    else :
        print("[INFO] Strat training " + args.which_net + " network on " + args.which_device)
        start = time.time()
        for epoch in range(args.train_epoch):  
            print("[INFO] Training " + str(epoch) + " epoch...")
            running_loss = 0.0
            start_0 = time.time()
            for i, data in tqdm(enumerate(trainloader, 0)):
                # 输入数据
                inputs, labels = data
                inputs = inputs.to(device)
                labels = labels.to(device)
                # 梯度清零
                optimizer.zero_grad()
                # 前向传播、计算损失、反向计算、参数更新
                outputs = net(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                # 打印日志
                running_loss += loss.item()
                if i % 2000 == 1999: # 每2000个batch打印一下训练状态
                    end_2000 = time.time()
                    print('[%d, %5d] loss: %.3f take %.5f s' \
                          % (epoch+1, i+1, running_loss / 2000, (end_2000-start_0)))
                    running_loss = 0.0
                    start_0 = time.time()
        end = time.time()
        print('Finished Training: ' + str(end- start) + 's')
        input_rand = torch.zeros((args.train_batch,3,args.image_size,args.image_size))
        net = net.to("cpu")
        torch.onnx.export(net, input_rand, args.which_net + '.onnx', input_names = ["image"], output_names = ["label"])

# 网络推理
# correct = 0 # 预测正确的图片数
# total = 0 # 总共的图片数
# 由于测试的时候不需要求导，可以暂时关闭autograd，提高速度，节约内存
# with torch.no_grad():
#     for data in testloader:
#         images, labels = data
#         images = images.to(device)
#         labels = labels.to(device)
#         outputs = net(images)
#         _, predicted = torch.max(outputs, 1)
#         total += labels.size(0)
#         correct += (predicted == labels).sum()
# print('10000张测试集中的准确率为: %f %%' % (100 * correct / total))

###################################################################################
# Gofusion 运行
if args.gofusion_infer == "True":
    from pyinfinitensor.onnx import OnnxStub, backend
    if len(args.onnx_file) != 0:
        model = onnx.load(args.onnx_file)
    else:
        model = onnx.load('./'+ args.which_net +'.onnx')
    model, check = simplify(model)
    eliminate_batchnorm(model)

    if args.which_device == "cpu":
        gofusion_model = OnnxStub(model, backend.cpu_runtime())
    elif args.which_device == "mlu":
        gofusion_model = OnnxStub(model, backend.bang_runtime())
    elif args.which_device == "cuda":
        gofusion_model = OnnxStub(model, backend.cuda_runtime())
    elif args.which_device == "xpu":
        gofusion_model = OnnxStub(model, backend.xpu_runtime())

    model = gofusion_model
    model.init()
    model.convert_nhwc()
    print("[INFO] Gofusion strat infer " + args.which_net + " network on " + args.which_device)
    correct = 0 # 预测正确的图片数
    total = 0 # 总共的图片数
    total_time = 0.0
    # 使用本项目的 Runtime 运行刚才加载并转换的模型, 验证是否一致
    for data in tqdm(testloader):
        images, labels = data
        next(model.inputs.items().__iter__())[1].copyin_float(images.reshape(-1).tolist())
        start_time = time.time()
        model.run()
        end_time = time.time()
        outputs = next(model.outputs.items().__iter__())[1].copyout_float()
        outputs = torch.tensor(outputs)
        outputs = torch.reshape(outputs,(args.infer_batch,10))
        total_time += (end_time - start_time)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum()
    print('%d 张测试的准确率为: %f %%' % (total, 100 * correct / total))
    print('BatchSize = %d, GoFusion 推理耗时 %f s' % (labels.size(0), total_time / (total / (labels.size(0)))))
    del model, gofusion_model

###################################################################################
# Pytorch 运行
# 将模型转换为对应版本
if args.pytorch_infer == "True":
    if args.which_device == "cpu":
        pass
    elif args.which_device == "mlu":
        import torch_mlu
    elif args.which_device == "cuda":
        pass
    elif args.which_device == "xpu":
        print("[INFO] Pytorch doesn't support " + args.which_device)
    if len(args.onnx_file) != 0:
        model = onnx.load(args.onnx_file)
    else:
        model = onnx.load('./'+ args.which_net +'.onnx')
    model, check = simplify(model)

    device = args.which_device

    target_version = 13
    converted_model = version_converter.convert_version(model, target_version)
    torch_model = convert(converted_model)
    torch_model.to(device)
    torch_model.eval()
    print("[INFO] Pytorch strat infer " + args.which_net + " network on " + args.which_device)
    correct = 0 # 预测正确的图片数
    total = 0 # 总共的图片数
    total_time = 0.0
    with torch.no_grad():
        for data in tqdm(testloader):
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            start_time = time.time()
            outputs = torch_model(images)
            end_time = time.time()
            total_time += (end_time - start_time)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum()
    print('%d 张测试的准确率为: %f %%' % (total, 100 * correct / total))
    print('BatchSize = %d, Pytorch 推理耗时 %f s' % (labels.size(0), total_time / (total / (labels.size(0)))))
    del torch_model, model

###################################################################################
# Magicmind 运行
# 将模型转换为对应版本
if args.magicmind_infer == "True":
    import magicmind.python.runtime as mm
    from magicmind.python.runtime import ModelKind, Network, Builder
    from magicmind.python.runtime.parser import Parser
    dev = mm.Device()
    dev.id = 0
    assert dev.active().ok()

    parser = Parser(ModelKind.kOnnx)
    network = Network()
    builder = Builder()
    if len(args.onnx_file) != 0:
        parser.parse(network, args.onnx_file)
    else:
        parser.parse(network, './' + args.which_net + '.onnx')

    model = builder.build_model("model", network) 

    engine = model.create_i_engine()
    context = engine.create_i_context()
    queue = dev.create_queue()
    inputs = context.create_inputs()

    print("[INFO] Magicmind strat infer " + args.which_net + " network on " + args.which_device)
    correct = 0 # 预测正确的图片数
    total = 0 # 总共的图片数
    total_time = 0.0
    for data in tqdm(testloader):
        images, labels = data
        images = images.numpy()
        inputs[0].from_numpy(images)
        outputs = context.create_outputs(inputs)
        start_time = time.time()
        context.enqueue(inputs, outputs, queue)
        queue.sync()
        end_time = time.time()
        outputs = outputs[0].asnumpy()
        outputs = torch.tensor(outputs)
        total_time += (end_time - start_time)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum()
    print('%d 张测试的准确率为: %f %%' % (total, 100 * correct / total))
    print('BatchSize = %d, Pytorch 推理耗时 %f s' % (labels.size(0), total_time / (total / (labels.size(0)))))

