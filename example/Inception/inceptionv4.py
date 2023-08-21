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
parser.add_argument("--which_device", type=str, default="cpu", choices=["mlu", "cuda", "cpu", "xpu"])
parser.add_argument("--pytorch_infer", type=str, default="False", choices=["False", "True"])
parser.add_argument("--gofusion_infer", type=str, default="False", choices=["False", "True"])
parser.add_argument("--magicmind_infer", type=str, default="False", choices=["False", "True"])
parser.add_argument("--onnx_file", type=str, default="")
parser.add_argument("--image_size", default="224", type=int, help="The width/height of image")
parser.add_argument("--sample", default="1", type=float, help="The percentage of the all images which to be infer.")
args = parser.parse_args()

print(vars(args))

# 定义网络并显示
class BasicConv2d(nn.Module):

    def __init__(self, input_channels, output_channels, **kwargs):
        super().__init__()
        self.conv = nn.Conv2d(input_channels, output_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(output_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)

        return x

class Inception_Stem(nn.Module):

    #"""Figure 3. The schema for stem of the pure Inception-v4 and
    #Inception-ResNet-v2 networks. This is the input part of those
    #networks."""
    def __init__(self, input_channels):
        super().__init__()
        self.conv1 = nn.Sequential(
            BasicConv2d(input_channels, 32, kernel_size=3),
            BasicConv2d(32, 32, kernel_size=3, padding=1),
            BasicConv2d(32, 64, kernel_size=3, padding=1)
        )

        self.branch3x3_conv = BasicConv2d(64, 96, kernel_size=3, padding=1)
        self.branch3x3_pool = nn.MaxPool2d(3, stride=1, padding=1)

        self.branch7x7a = nn.Sequential(
            BasicConv2d(160, 64, kernel_size=1),
            BasicConv2d(64, 64, kernel_size=(7, 1), padding=(3, 0)),
            BasicConv2d(64, 64, kernel_size=(1, 7), padding=(0, 3)),
            BasicConv2d(64, 96, kernel_size=3, padding=1)
        )

        self.branch7x7b = nn.Sequential(
            BasicConv2d(160, 64, kernel_size=1),
            BasicConv2d(64, 96, kernel_size=3, padding=1)
        )

        self.branchpoola = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.branchpoolb = BasicConv2d(192, 192, kernel_size=3, stride=1, padding=1)

    def forward(self, x):

        x = self.conv1(x)

        x = [
            self.branch3x3_conv(x),
            self.branch3x3_pool(x)
        ]
        x = torch.cat(x, 1)

        x = [
            self.branch7x7a(x),
            self.branch7x7b(x)
        ]
        x = torch.cat(x, 1)

        x = [
            self.branchpoola(x),
            self.branchpoolb(x)
        ]

        x = torch.cat(x, 1)

        return x

class InceptionA(nn.Module):

    #"""Figure 4. The schema for 35 × 35 grid modules of the pure
    #Inception-v4 network. This is the Inception-A block of Figure 9."""
    def __init__(self, input_channels):
        super().__init__()

        self.branch3x3stack = nn.Sequential(
            BasicConv2d(input_channels, 64, kernel_size=1),
            BasicConv2d(64, 96, kernel_size=3, padding=1),
            BasicConv2d(96, 96, kernel_size=3, padding=1)
        )

        self.branch3x3 = nn.Sequential(
            BasicConv2d(input_channels, 64, kernel_size=1),
            BasicConv2d(64, 96, kernel_size=3, padding=1)
        )

        self.branch1x1 = BasicConv2d(input_channels, 96, kernel_size=1)

        self.branchpool = nn.Sequential(
            nn.AvgPool2d(kernel_size=3, stride=1, padding=1),
            BasicConv2d(input_channels, 96, kernel_size=1)
        )

    def forward(self, x):

        x = [
            self.branch3x3stack(x),
            self.branch3x3(x),
            self.branch1x1(x),
            self.branchpool(x)
        ]

        return torch.cat(x, 1)

class ReductionA(nn.Module):

    #"""Figure 7. The schema for 35 × 35 to 17 × 17 reduction module.
    #Different variants of this blocks (with various number of filters)
    #are used in Figure 9, and 15 in each of the new Inception(-v4, - ResNet-v1,
    #-ResNet-v2) variants presented in this paper. The k, l, m, n numbers
    #represent filter bank sizes which can be looked up in Table 1.
    def __init__(self, input_channels, k, l, m, n):

        super().__init__()
        self.branch3x3stack = nn.Sequential(
            BasicConv2d(input_channels, k, kernel_size=1),
            BasicConv2d(k, l, kernel_size=3, padding=1),
            BasicConv2d(l, m, kernel_size=3, stride=2)
        )

        self.branch3x3 = BasicConv2d(input_channels, n, kernel_size=3, stride=2)
        self.branchpool = nn.MaxPool2d(kernel_size=3, stride=2)
        self.output_channels = input_channels + n + m

    def forward(self, x):

        x = [
            self.branch3x3stack(x),
            self.branch3x3(x),
            self.branchpool(x)
        ]

        return torch.cat(x, 1)

class InceptionB(nn.Module):

    #"""Figure 5. The schema for 17 × 17 grid modules of the pure Inception-v4 network.
    #This is the Inception-B block of Figure 9."""
    def __init__(self, input_channels):
        super().__init__()

        self.branch7x7stack = nn.Sequential(
            BasicConv2d(input_channels, 192, kernel_size=1),
            BasicConv2d(192, 192, kernel_size=(1, 7), padding=(0, 3)),
            BasicConv2d(192, 224, kernel_size=(7, 1), padding=(3, 0)),
            BasicConv2d(224, 224, kernel_size=(1, 7), padding=(0, 3)),
            BasicConv2d(224, 256, kernel_size=(7, 1), padding=(3, 0))
        )

        self.branch7x7 = nn.Sequential(
            BasicConv2d(input_channels, 192, kernel_size=1),
            BasicConv2d(192, 224, kernel_size=(1, 7), padding=(0, 3)),
            BasicConv2d(224, 256, kernel_size=(7, 1), padding=(3, 0))
        )

        self.branch1x1 = BasicConv2d(input_channels, 384, kernel_size=1)

        self.branchpool = nn.Sequential(
            nn.AvgPool2d(3, stride=1, padding=1),
            BasicConv2d(input_channels, 128, kernel_size=1)
        )

    def forward(self, x):
        x = [
            self.branch1x1(x),
            self.branch7x7(x),
            self.branch7x7stack(x),
            self.branchpool(x)
        ]

        return torch.cat(x, 1)

class ReductionB(nn.Module):

    #"""Figure 8. The schema for 17 × 17 to 8 × 8 grid-reduction mod- ule.
    #This is the reduction module used by the pure Inception-v4 network in
    #Figure 9."""
    def __init__(self, input_channels):

        super().__init__()
        self.branch7x7 = nn.Sequential(
            BasicConv2d(input_channels, 256, kernel_size=1),
            BasicConv2d(256, 256, kernel_size=(1, 7), padding=(0, 3)),
            BasicConv2d(256, 320, kernel_size=(7, 1), padding=(3, 0)),
            BasicConv2d(320, 320, kernel_size=3, stride=2, padding=1)
        )

        self.branch3x3 = nn.Sequential(
            BasicConv2d(input_channels, 192, kernel_size=1),
            BasicConv2d(192, 192, kernel_size=3, stride=2, padding=1)
        )

        self.branchpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):

        x = [
            self.branch3x3(x),
            self.branch7x7(x),
            self.branchpool(x)
        ]

        return torch.cat(x, 1)

class InceptionC(nn.Module):

    def __init__(self, input_channels):
        #"""Figure 6. The schema for 8×8 grid modules of the pure
        #Inceptionv4 network. This is the Inception-C block of Figure 9."""

        super().__init__()

        self.branch3x3stack = nn.Sequential(
            BasicConv2d(input_channels, 384, kernel_size=1),
            BasicConv2d(384, 448, kernel_size=(1, 3), padding=(0, 1)),
            BasicConv2d(448, 512, kernel_size=(3, 1), padding=(1, 0)),
        )
        self.branch3x3stacka = BasicConv2d(512, 256, kernel_size=(1, 3), padding=(0, 1))
        self.branch3x3stackb = BasicConv2d(512, 256, kernel_size=(3, 1), padding=(1, 0))

        self.branch3x3 = BasicConv2d(input_channels, 384, kernel_size=1)
        self.branch3x3a = BasicConv2d(384, 256, kernel_size=(3, 1), padding=(1, 0))
        self.branch3x3b = BasicConv2d(384, 256, kernel_size=(1, 3), padding=(0, 1))

        self.branch1x1 = BasicConv2d(input_channels, 256, kernel_size=1)

        self.branchpool = nn.Sequential(
            nn.AvgPool2d(kernel_size=3, stride=1, padding=1),
            BasicConv2d(input_channels, 256, kernel_size=1)
        )

    def forward(self, x):
        branch3x3stack_output = self.branch3x3stack(x)
        branch3x3stack_output = [
            self.branch3x3stacka(branch3x3stack_output),
            self.branch3x3stackb(branch3x3stack_output)
        ]
        branch3x3stack_output = torch.cat(branch3x3stack_output, 1)

        branch3x3_output = self.branch3x3(x)
        branch3x3_output = [
            self.branch3x3a(branch3x3_output),
            self.branch3x3b(branch3x3_output)
        ]
        branch3x3_output = torch.cat(branch3x3_output, 1)

        branch1x1_output = self.branch1x1(x)

        branchpool = self.branchpool(x)

        output = [
            branch1x1_output,
            branch3x3_output,
            branch3x3stack_output,
            branchpool
        ]

        return torch.cat(output, 1)

class InceptionV4(nn.Module):

    def __init__(self, A, B, C, k=192, l=224, m=256, n=384, class_nums=100):

        super().__init__()
        self.stem = Inception_Stem(3)
        self.inception_a = self._generate_inception_module(384, 384, A, InceptionA)
        self.reduction_a = ReductionA(384, k, l, m, n)
        output_channels = self.reduction_a.output_channels
        self.inception_b = self._generate_inception_module(output_channels, 1024, B, InceptionB)
        self.reduction_b = ReductionB(1024)
        self.inception_c = self._generate_inception_module(1536, 1536, C, InceptionC)
        self.avgpool = nn.AvgPool2d(7)

        #"""Dropout (keep 0.8)"""
        self.dropout = nn.Dropout2d(1 - 0.8)
        self.linear = nn.Linear(1536, class_nums)

    def forward(self, x):
        x = self.stem(x)
        x = self.inception_a(x)
        x = self.reduction_a(x)
        x = self.inception_b(x)
        x = self.reduction_b(x)
        x = self.inception_c(x)
        x = self.avgpool(x)
        x = self.dropout(x)
        x = x.view(-1, 1536)
        x = self.linear(x)

        return x

    @staticmethod
    def _generate_inception_module(input_channels, output_channels, block_num, block):

        layers = nn.Sequential()
        for l in range(block_num):
            layers.add_module("{}_{}".format(block.__name__, l), block(input_channels))
            input_channels = output_channels

        return layers

class InceptionResNetA(nn.Module):

    #"""Figure 16. The schema for 35 × 35 grid (Inception-ResNet-A)
    #module of the Inception-ResNet-v2 network."""
    def __init__(self, input_channels):

        super().__init__()
        self.branch3x3stack = nn.Sequential(
            BasicConv2d(input_channels, 32, kernel_size=1),
            BasicConv2d(32, 48, kernel_size=3, padding=1),
            BasicConv2d(48, 64, kernel_size=3, padding=1)
        )

        self.branch3x3 = nn.Sequential(
            BasicConv2d(input_channels, 32, kernel_size=1),
            BasicConv2d(32, 32, kernel_size=3, padding=1)
        )

        self.branch1x1 = BasicConv2d(input_channels, 32, kernel_size=1)

        self.reduction1x1 = nn.Conv2d(128, 384, kernel_size=1)
        self.shortcut = nn.Conv2d(input_channels, 384, kernel_size=1)
        self.bn = nn.BatchNorm2d(384)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):

        residual = [
            self.branch1x1(x),
            self.branch3x3(x),
            self.branch3x3stack(x)
        ]

        residual = torch.cat(residual, 1)
        residual = self.reduction1x1(residual)
        shortcut = self.shortcut(x)

        output = self.bn(shortcut + residual)
        output = self.relu(output)

        return output

class InceptionResNetB(nn.Module):

    #"""Figure 17. The schema for 17 × 17 grid (Inception-ResNet-B) module of
    #the Inception-ResNet-v2 network."""
    def __init__(self, input_channels):

        super().__init__()
        self.branch7x7 = nn.Sequential(
            BasicConv2d(input_channels, 128, kernel_size=1),
            BasicConv2d(128, 160, kernel_size=(1, 7), padding=(0, 3)),
            BasicConv2d(160, 192, kernel_size=(7, 1), padding=(3, 0))
        )

        self.branch1x1 = BasicConv2d(input_channels, 192, kernel_size=1)

        self.reduction1x1 = nn.Conv2d(384, 1154, kernel_size=1)
        self.shortcut = nn.Conv2d(input_channels, 1154, kernel_size=1)

        self.bn = nn.BatchNorm2d(1154)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = [
            self.branch1x1(x),
            self.branch7x7(x)
        ]

        residual = torch.cat(residual, 1)

        #"""In general we picked some scaling factors between 0.1 and 0.3 to scale the residuals
        #before their being added to the accumulated layer activations (cf. Figure 20)."""
        residual = self.reduction1x1(residual) * 0.1

        shortcut = self.shortcut(x)

        output = self.bn(residual + shortcut)
        output = self.relu(output)

        return output


class InceptionResNetC(nn.Module):

    def __init__(self, input_channels):

        #Figure 19. The schema for 8×8 grid (Inception-ResNet-C)
        #module of the Inception-ResNet-v2 network."""
        super().__init__()
        self.branch3x3 = nn.Sequential(
            BasicConv2d(input_channels, 192, kernel_size=1),
            BasicConv2d(192, 224, kernel_size=(1, 3), padding=(0, 1)),
            BasicConv2d(224, 256, kernel_size=(3, 1), padding=(1, 0))
        )

        self.branch1x1 = BasicConv2d(input_channels, 192, kernel_size=1)
        self.reduction1x1 = nn.Conv2d(448, 2048, kernel_size=1)
        self.shorcut = nn.Conv2d(input_channels, 2048, kernel_size=1)
        self.bn = nn.BatchNorm2d(2048)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = [
            self.branch1x1(x),
            self.branch3x3(x)
        ]

        residual = torch.cat(residual, 1)
        residual = self.reduction1x1(residual) * 0.1

        shorcut = self.shorcut(x)

        output = self.bn(shorcut + residual)
        output = self.relu(output)

        return output

class InceptionResNetReductionA(nn.Module):

    #"""Figure 7. The schema for 35 × 35 to 17 × 17 reduction module.
    #Different variants of this blocks (with various number of filters)
    #are used in Figure 9, and 15 in each of the new Inception(-v4, - ResNet-v1,
    #-ResNet-v2) variants presented in this paper. The k, l, m, n numbers
    #represent filter bank sizes which can be looked up in Table 1.
    def __init__(self, input_channels, k, l, m, n):

        super().__init__()
        self.branch3x3stack = nn.Sequential(
            BasicConv2d(input_channels, k, kernel_size=1),
            BasicConv2d(k, l, kernel_size=3, padding=1),
            BasicConv2d(l, m, kernel_size=3, stride=2)
        )

        self.branch3x3 = BasicConv2d(input_channels, n, kernel_size=3, stride=2)
        self.branchpool = nn.MaxPool2d(kernel_size=3, stride=2)
        self.output_channels = input_channels + n + m

    def forward(self, x):

        x = [
            self.branch3x3stack(x),
            self.branch3x3(x),
            self.branchpool(x)
        ]

        return torch.cat(x, 1)

class InceptionResNetReductionB(nn.Module):

    #"""Figure 18. The schema for 17 × 17 to 8 × 8 grid-reduction module.
    #Reduction-B module used by the wider Inception-ResNet-v1 network in
    #Figure 15."""
    #I believe it was a typo(Inception-ResNet-v1 should be Inception-ResNet-v2)
    def __init__(self, input_channels):

        super().__init__()
        self.branchpool = nn.MaxPool2d(3, stride=2)

        self.branch3x3a = nn.Sequential(
            BasicConv2d(input_channels, 256, kernel_size=1),
            BasicConv2d(256, 384, kernel_size=3, stride=2)
        )

        self.branch3x3b = nn.Sequential(
            BasicConv2d(input_channels, 256, kernel_size=1),
            BasicConv2d(256, 288, kernel_size=3, stride=2)
        )

        self.branch3x3stack = nn.Sequential(
            BasicConv2d(input_channels, 256, kernel_size=1),
            BasicConv2d(256, 288, kernel_size=3, padding=1),
            BasicConv2d(288, 320, kernel_size=3, stride=2)
        )

    def forward(self, x):
        x = [
            self.branch3x3a(x),
            self.branch3x3b(x),
            self.branch3x3stack(x),
            self.branchpool(x)
        ]

        x = torch.cat(x, 1)
        return x

class InceptionResNetV2(nn.Module):

    def __init__(self, A, B, C, k=256, l=256, m=384, n=384, class_nums=100):
        super().__init__()
        self.stem = Inception_Stem(3)
        self.inception_resnet_a = self._generate_inception_module(384, 384, A, InceptionResNetA)
        self.reduction_a = InceptionResNetReductionA(384, k, l, m, n)
        output_channels = self.reduction_a.output_channels
        self.inception_resnet_b = self._generate_inception_module(output_channels, 1154, B, InceptionResNetB)
        self.reduction_b = InceptionResNetReductionB(1154)
        self.inception_resnet_c = self._generate_inception_module(2146, 2048, C, InceptionResNetC)

        #6x6 featuresize
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        #"""Dropout (keep 0.8)"""
        self.dropout = nn.Dropout2d(1 - 0.8)
        self.linear = nn.Linear(2048, class_nums)

    def forward(self, x):
        x = self.stem(x)
        x = self.inception_resnet_a(x)
        x = self.reduction_a(x)
        x = self.inception_resnet_b(x)
        x = self.reduction_b(x)
        x = self.inception_resnet_c(x)
        x = self.avgpool(x)
        x = self.dropout(x)
        x = x.view(-1, 2048)
        x = self.linear(x)

        return x

    @staticmethod
    def _generate_inception_module(input_channels, output_channels, block_num, block):

        layers = nn.Sequential()
        for l in range(block_num):
            layers.add_module("{}_{}".format(block.__name__, l), block(input_channels))
            input_channels = output_channels

        return layers

net = InceptionV4(4, 7, 3, class_nums = 10)
#print(net)

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

# 定义损失函数与优化器
criterion = nn.CrossEntropyLoss() # 交叉熵损失函数
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
# 定义硬件设备
input_rand = torch.zeros((args.infer_batch,3,args.image_size,args.image_size))
torch.onnx.export(net, input_rand, 'inceptionv4' + '_untrained' + '.onnx', input_names = ["image"], output_names = ["label"])

# 网络训练
if args.train_epoch != 0:
    device = torch.device(args.which_device)
    net = net.to(device)
    if args.which_device == "mlu" or args.which_device == "xpu" :
        print("[INFO] Pytorch doesn't support network train on " + args.which_device)
    else :
        print("[INFO] Strat training " + "inceptionv4" + " network on " + args.which_device)
        start = time.time()
        for epoch in range(args.train_epoch):  
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
        torch.onnx.export(net, input_rand, 'inceptionv4' + '.onnx', input_names = ["image"], output_names = ["label"])
    
# # 网络推理
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

# Gofusion 运行
if args.gofusion_infer == "True":
    if len(args.onnx_file) != 0:
        model = onnx.load(args.onnx_file)
    else:
        model = onnx.load('./'+ 'inceptionv4' +'.onnx')
    model, check = simplify(model)

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
    print("[INFO] Gofusion strat infer " + 'inceptionv4' + " network on " + args.which_device)
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
        model = onnx.load('./'+ 'inceptionv4' +'.onnx')
    model, check = simplify(model)

    target_version = 13
    converted_model = version_converter.convert_version(model, target_version)
    torch_model = convert(converted_model)
    torch_model.to(args.which_device)
    torch_model.eval()
    print("[INFO] Pytorch strat infer " + "inceptionv4" + " network on " + args.which_device)
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

    print("[INFO] Magicmind strat infer inceptionv4 network on " + args.which_device)
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
