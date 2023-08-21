# 引入 torch 包、神经网络包、函数包、优化器
import torch
import torch.nn as nn
import torch.nn.functional as func
import torch.optim as optim
# 引入图像包、图像处理包、显示包、时间包
import torchvision
import torchvision.transforms as transforms
from torchvision.transforms import ToPILImage
from torch.utils.data import Sampler
import random
import time
# 引入onnx 相关工具
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
parser.add_argument("--which_net", type=str, required=True, choices=["densenet121", "densenet169", "densenet201", "densenet161"])
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
class Bottleneck(nn.Module):
    def __init__(self, in_channels, growth_rate):
        super().__init__()
        #"""In  our experiments, we let each 1×1 convolution
        #produce 4k feature-maps."""
        inner_channel = 4 * growth_rate

        #"""We find this design especially effective for DenseNet and
        #we refer to our network with such a bottleneck layer, i.e.,
        #to the BN-ReLU-Conv(1×1)-BN-ReLU-Conv(3×3) version of H ` ,
        #as DenseNet-B."""
        self.bottle_neck = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, inner_channel, kernel_size=1, bias=False),
            nn.BatchNorm2d(inner_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(inner_channel, growth_rate, kernel_size=3, padding=1, bias=False)
        )

    def forward(self, x):
        return torch.cat([x, self.bottle_neck(x)], 1)

#"""We refer to layers between blocks as transition
#layers, which do convolution and pooling."""
class Transition(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        #"""The transition layers used in our experiments
        #consist of a batch normalization layer and an 1×1
        #convolutional layer followed by a 2×2 average pooling
        #layer""".
        self.down_sample = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.AvgPool2d(2, stride=2)
        )

    def forward(self, x):
        return self.down_sample(x)

#DesneNet-BC
#B stands for bottleneck layer(BN-RELU-CONV(1x1)-BN-RELU-CONV(3x3))
#C stands for compression factor(0<=theta<=1)
class DenseNet(nn.Module):
    def __init__(self, block, nblocks, growth_rate=12, reduction=0.5, num_class=100):
        super().__init__()
        self.growth_rate = growth_rate

        #"""Before entering the first dense block, a convolution
        #with 16 (or twice the growth rate for DenseNet-BC)
        #output channels is performed on the input images."""
        inner_channels = 2 * growth_rate

        #For convolutional layers with kernel size 3×3, each
        #side of the inputs is zero-padded by one pixel to keep
        #the feature-map size fixed.
        self.conv1 = nn.Conv2d(3, inner_channels, kernel_size=3, padding=1, bias=False)

        self.features = nn.Sequential()

        for index in range(len(nblocks) - 1):
            self.features.add_module("dense_block_layer_{}".format(index), self._make_dense_layers(block, inner_channels, nblocks[index]))
            inner_channels += growth_rate * nblocks[index]

            #"""If a dense block contains m feature-maps, we let the
            #following transition layer generate θm output feature-
            #maps, where 0 < θ ≤ 1 is referred to as the compression
            #fac-tor.
            out_channels = int(reduction * inner_channels) # int() will automatic floor the value
            self.features.add_module("transition_layer_{}".format(index), Transition(inner_channels, out_channels))
            inner_channels = out_channels

        self.features.add_module("dense_block{}".format(len(nblocks) - 1), self._make_dense_layers(block, inner_channels, nblocks[len(nblocks)-1]))
        inner_channels += growth_rate * nblocks[len(nblocks) - 1]
        self.features.add_module('bn', nn.BatchNorm2d(inner_channels))
        self.features.add_module('relu', nn.ReLU(inplace=True))

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.linear = nn.Linear(inner_channels, num_class)

    def forward(self, x):
        output = self.conv1(x)
        output = self.features(output)
        output = self.avgpool(output)
        output = output.view(output.size()[0], -1)
        output = self.linear(output)
        return output

    def _make_dense_layers(self, block, in_channels, nblocks):
        dense_block = nn.Sequential()
        for index in range(nblocks):
            dense_block.add_module('bottle_neck_layer_{}'.format(index), block(in_channels, self.growth_rate))
            in_channels += self.growth_rate
        return dense_block

def densenet121():
    return DenseNet(Bottleneck, [6,12,24,16], growth_rate=32, num_class=10)

def densenet169():
    return DenseNet(Bottleneck, [6,12,32,32], growth_rate=32, num_class=10)

def densenet201():
    return DenseNet(Bottleneck, [6,12,48,32], growth_rate=32, num_class=10)

def densenet161():
    return DenseNet(Bottleneck, [6,12,36,24], growth_rate=48, num_class=10)

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
        transforms.Resize((args.image_size, args.image_size)),
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
if args.which_net == "densenet121":
    net = densenet121()
elif args.which_net == "densenet169":
    net = densenet169()
elif args.which_net == "densenet201":
    net = densenet201()
elif args.which_net == "densenet161":
    net = densenet161()
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
torch.onnx.export(evalnet, input_rand, args.which_net + '_untrained' + '.onnx', input_names = ["image"], output_names = ["label"], do_constant_folding=True)

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
        torch.onnx.export(net, input_rand, args.which_net + '.onnx', input_names = ["image"], output_names = ["label"])

# 网络推理
# correct = 0 # 预测正确的图片数
# total = 0 # 总共的图片数
# # 由于测试的时候不需要求导，可以暂时关闭autograd，提高速度，节约内存
# with torch.no_grad():
#     for data in testloader:
#         images, labels = data
#         images = images.to(device)
#         labels = labels.to(device)
#         outputs = net(images)
#         _, predicted = torch.max(outputs, 1)
#         total += labels.size(0)
#         correct += (predicted == labels).sum()
# 
# print('10000张测试集中的准确率为: %f %%' % (100 * correct / total))

###################################################################################
# Gofusion 运行
if args.gofusion_infer == "True":
    if len(args.onnx_file) != 0:
        model = onnx.load(args.onnx_file)
    else:
        model = onnx.load('./'+ args.which_net +'.onnx')
    model, check = simplify(model)
    eliminate_batchnorm(model)

    from pyinfinitensor.onnx import OnnxStub, backend
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

    target_version = 13
    converted_model = version_converter.convert_version(model, target_version)
    torch_model = convert(converted_model)
    torch_model.to(args.which_device)
    torch_model.eval()
    print("[INFO] Pytorch strat infer " + args.which_net + " network on " + args.which_device)
    correct = 0 # 预测正确的图片数
    total = 0 # 总共的图片数
    total_time = 0.0
    with torch.no_grad():
        for data in tqdm(testloader):
            images, labels = data
            images = images.to(args.which_device)
            labels = labels.to(args.which_device)
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
