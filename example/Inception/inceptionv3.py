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

#same naive inception module
class InceptionA(nn.Module):

    def __init__(self, input_channels, pool_features):
        super().__init__()
        self.branch1x1 = BasicConv2d(input_channels, 64, kernel_size=1)

        self.branch5x5 = nn.Sequential(
            BasicConv2d(input_channels, 48, kernel_size=1),
            BasicConv2d(48, 64, kernel_size=5, padding=2)
        )

        self.branch3x3 = nn.Sequential(
            BasicConv2d(input_channels, 64, kernel_size=1),
            BasicConv2d(64, 96, kernel_size=3, padding=1),
            BasicConv2d(96, 96, kernel_size=3, padding=1)
        )

        self.branchpool = nn.Sequential(
            nn.AvgPool2d(kernel_size=3, stride=1, padding=1),
            BasicConv2d(input_channels, pool_features, kernel_size=3, padding=1)
        )

    def forward(self, x):

        #x -> 1x1(same)
        branch1x1 = self.branch1x1(x)

        #x -> 1x1 -> 5x5(same)
        branch5x5 = self.branch5x5(x)
        #branch5x5 = self.branch5x5_2(branch5x5)

        #x -> 1x1 -> 3x3 -> 3x3(same)
        branch3x3 = self.branch3x3(x)

        #x -> pool -> 1x1(same)
        branchpool = self.branchpool(x)

        outputs = [branch1x1, branch5x5, branch3x3, branchpool]

        return torch.cat(outputs, 1)

#downsample
#Factorization into smaller convolutions
class InceptionB(nn.Module):

    def __init__(self, input_channels):
        super().__init__()

        self.branch3x3 = BasicConv2d(input_channels, 384, kernel_size=3, stride=2)

        self.branch3x3stack = nn.Sequential(
            BasicConv2d(input_channels, 64, kernel_size=1),
            BasicConv2d(64, 96, kernel_size=3, padding=1),
            BasicConv2d(96, 96, kernel_size=3, stride=2)
        )

        self.branchpool = nn.MaxPool2d(kernel_size=3, stride=2)

    def forward(self, x):

        #x - > 3x3(downsample)
        branch3x3 = self.branch3x3(x)

        #x -> 3x3 -> 3x3(downsample)
        branch3x3stack = self.branch3x3stack(x)

        #x -> avgpool(downsample)
        branchpool = self.branchpool(x)

        #"""We can use two parallel stride 2 blocks: P and C. P is a pooling
        #layer (either average or maximum pooling) the activation, both of
        #them are stride 2 the filter banks of which are concatenated as in
        #figure 10."""
        outputs = [branch3x3, branch3x3stack, branchpool]

        return torch.cat(outputs, 1)

#Factorizing Convolutions with Large Filter Size
class InceptionC(nn.Module):
    def __init__(self, input_channels, channels_7x7):
        super().__init__()
        self.branch1x1 = BasicConv2d(input_channels, 192, kernel_size=1)

        c7 = channels_7x7

        #In theory, we could go even further and argue that one can replace any n × n
        #convolution by a 1 × n convolution followed by a n × 1 convolution and the
        #computational cost saving increases dramatically as n grows (see figure 6).
        self.branch7x7 = nn.Sequential(
            BasicConv2d(input_channels, c7, kernel_size=1),
            BasicConv2d(c7, c7, kernel_size=(7, 1), padding=(3, 0)),
            BasicConv2d(c7, 192, kernel_size=(1, 7), padding=(0, 3))
        )

        self.branch7x7stack = nn.Sequential(
            BasicConv2d(input_channels, c7, kernel_size=1),
            BasicConv2d(c7, c7, kernel_size=(7, 1), padding=(3, 0)),
            BasicConv2d(c7, c7, kernel_size=(1, 7), padding=(0, 3)),
            BasicConv2d(c7, c7, kernel_size=(7, 1), padding=(3, 0)),
            BasicConv2d(c7, 192, kernel_size=(1, 7), padding=(0, 3))
        )

        self.branch_pool = nn.Sequential(
            nn.AvgPool2d(kernel_size=3, stride=1, padding=1),
            BasicConv2d(input_channels, 192, kernel_size=1),
        )

    def forward(self, x):

        #x -> 1x1(same)
        branch1x1 = self.branch1x1(x)

        #x -> 1layer 1*7 and 7*1 (same)
        branch7x7 = self.branch7x7(x)

        #x-> 2layer 1*7 and 7*1(same)
        branch7x7stack = self.branch7x7stack(x)

        #x-> avgpool (same)
        branchpool = self.branch_pool(x)

        outputs = [branch1x1, branch7x7, branch7x7stack, branchpool]

        return torch.cat(outputs, 1)

class InceptionD(nn.Module):

    def __init__(self, input_channels):
        super().__init__()

        self.branch3x3 = nn.Sequential(
            BasicConv2d(input_channels, 192, kernel_size=1),
            BasicConv2d(192, 320, kernel_size=3, stride=2)
        )

        self.branch7x7 = nn.Sequential(
            BasicConv2d(input_channels, 192, kernel_size=1),
            BasicConv2d(192, 192, kernel_size=(1, 7), padding=(0, 3)),
            BasicConv2d(192, 192, kernel_size=(7, 1), padding=(3, 0)),
            BasicConv2d(192, 192, kernel_size=3, stride=2)
        )

        self.branchpool = nn.AvgPool2d(kernel_size=3, stride=2)

    def forward(self, x):

        #x -> 1x1 -> 3x3(downsample)
        branch3x3 = self.branch3x3(x)

        #x -> 1x1 -> 1x7 -> 7x1 -> 3x3 (downsample)
        branch7x7 = self.branch7x7(x)

        #x -> avgpool (downsample)
        branchpool = self.branchpool(x)

        outputs = [branch3x3, branch7x7, branchpool]

        return torch.cat(outputs, 1)


#same
class InceptionE(nn.Module):
    def __init__(self, input_channels):
        super().__init__()
        self.branch1x1 = BasicConv2d(input_channels, 320, kernel_size=1)

        self.branch3x3_1 = BasicConv2d(input_channels, 384, kernel_size=1)
        self.branch3x3_2a = BasicConv2d(384, 384, kernel_size=(1, 3), padding=(0, 1))
        self.branch3x3_2b = BasicConv2d(384, 384, kernel_size=(3, 1), padding=(1, 0))

        self.branch3x3stack_1 = BasicConv2d(input_channels, 448, kernel_size=1)
        self.branch3x3stack_2 = BasicConv2d(448, 384, kernel_size=3, padding=1)
        self.branch3x3stack_3a = BasicConv2d(384, 384, kernel_size=(1, 3), padding=(0, 1))
        self.branch3x3stack_3b = BasicConv2d(384, 384, kernel_size=(3, 1), padding=(1, 0))

        self.branch_pool = nn.Sequential(
            nn.AvgPool2d(kernel_size=3, stride=1, padding=1),
            BasicConv2d(input_channels, 192, kernel_size=1)
        )

    def forward(self, x):

        #x -> 1x1 (same)
        branch1x1 = self.branch1x1(x)

        # x -> 1x1 -> 3x1
        # x -> 1x1 -> 1x3
        # concatenate(3x1, 1x3)
        #"""7. Inception modules with expanded the filter bank outputs.
        #This architecture is used on the coarsest (8 × 8) grids to promote
        #high dimensional representations, as suggested by principle
        #2 of Section 2."""
        branch3x3 = self.branch3x3_1(x)
        branch3x3 = [
            self.branch3x3_2a(branch3x3),
            self.branch3x3_2b(branch3x3)
        ]
        branch3x3 = torch.cat(branch3x3, 1)

        # x -> 1x1 -> 3x3 -> 1x3
        # x -> 1x1 -> 3x3 -> 3x1
        #concatenate(1x3, 3x1)
        branch3x3stack = self.branch3x3stack_1(x)
        branch3x3stack = self.branch3x3stack_2(branch3x3stack)
        branch3x3stack = [
            self.branch3x3stack_3a(branch3x3stack),
            self.branch3x3stack_3b(branch3x3stack)
        ]
        branch3x3stack = torch.cat(branch3x3stack, 1)

        branchpool = self.branch_pool(x)

        outputs = [branch1x1, branch3x3, branch3x3stack, branchpool]

        return torch.cat(outputs, 1)

class InceptionV3(nn.Module):

    def __init__(self, num_classes=100):
        super().__init__()
        self.Conv2d_1a_3x3 = BasicConv2d(3, 32, kernel_size=3, padding=1)
        self.Conv2d_2a_3x3 = BasicConv2d(32, 32, kernel_size=3, padding=1)
        self.Conv2d_2b_3x3 = BasicConv2d(32, 64, kernel_size=3, padding=1)
        self.Conv2d_3b_1x1 = BasicConv2d(64, 80, kernel_size=1)
        self.Conv2d_4a_3x3 = BasicConv2d(80, 192, kernel_size=3)

        #naive inception module
        self.Mixed_5b = InceptionA(192, pool_features=32)
        self.Mixed_5c = InceptionA(256, pool_features=64)
        self.Mixed_5d = InceptionA(288, pool_features=64)

        #downsample
        self.Mixed_6a = InceptionB(288)

        self.Mixed_6b = InceptionC(768, channels_7x7=128)
        self.Mixed_6c = InceptionC(768, channels_7x7=160)
        self.Mixed_6d = InceptionC(768, channels_7x7=160)
        self.Mixed_6e = InceptionC(768, channels_7x7=192)

        #downsample
        self.Mixed_7a = InceptionD(768)

        self.Mixed_7b = InceptionE(1280)
        self.Mixed_7c = InceptionE(2048)

        #6*6 feature size
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout2d()
        self.linear = nn.Linear(2048, num_classes)

    def forward(self, x):

        #32 -> 30
        x = self.Conv2d_1a_3x3(x)
        x = self.Conv2d_2a_3x3(x)
        x = self.Conv2d_2b_3x3(x)
        x = self.Conv2d_3b_1x1(x)
        x = self.Conv2d_4a_3x3(x)

        #30 -> 30
        x = self.Mixed_5b(x)
        x = self.Mixed_5c(x)
        x = self.Mixed_5d(x)

        #30 -> 14
        #Efficient Grid Size Reduction to avoid representation
        #bottleneck
        x = self.Mixed_6a(x)

        #14 -> 14
        #"""In practice, we have found that employing this factorization does not
        #work well on early layers, but it gives very good results on medium
        #grid-sizes (On m × m feature maps, where m ranges between 12 and 20).
        #On that level, very good results can be achieved by using 1 × 7 convolutions
        #followed by 7 × 1 convolutions."""
        x = self.Mixed_6b(x)
        x = self.Mixed_6c(x)
        x = self.Mixed_6d(x)
        x = self.Mixed_6e(x)

        #14 -> 6
        #Efficient Grid Size Reduction
        x = self.Mixed_7a(x)

        #6 -> 6
        #We are using this solution only on the coarsest grid,
        #since that is the place where producing high dimensional
        #sparse representation is the most critical as the ratio of
        #local processing (by 1 × 1 convolutions) is increased compared
        #to the spatial aggregation."""
        x = self.Mixed_7b(x)
        x = self.Mixed_7c(x)

        #6 -> 1
        x = self.avgpool(x)
        x = self.dropout(x)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        return x
net = InceptionV3(10)
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
                    sampler=sampler)

# 定义损失函数与优化器
criterion = nn.CrossEntropyLoss() # 交叉熵损失函数
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
# 定义硬件设备
input_rand = torch.zeros((args.infer_batch,3,args.image_size,args.image_size))
torch.onnx.export(net, input_rand, 'inceptionv3' + '_untrained' + '.onnx', input_names = ["image"], output_names = ["label"])

# 网络训练
if args.train_epoch != 0:
    device = torch.device(args.which_device)
    net = net.to(device)
    if args.which_device == "mlu" or args.which_device == "xpu" :
        print("[INFO] Pytorch doesn't support network train on " + args.which_device)
    else :
        print("[INFO] Strat training " + "inceptionv3" + " network on " + args.which_device)
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
        input_rand = torch.zeros((args.train_batch, 3, args.image_size, args.image_size))
        net = net.to("cpu")
        torch.onnx.export(net, input_rand, 'inceptionv3' + '.onnx', input_names = ["image"], output_names = ["label"])

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

###################################################################################
# Gofusion 运行
if args.gofusion_infer == "True":
    if len(args.onnx_file) != 0:
        model = onnx.load(args.onnx_file)
    else:
        model = onnx.load('./'+ 'inceptionv3' +'.onnx')
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
    print("[INFO] Gofusion strat infer " + 'inceptionv3' + " network on " + args.which_device)
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
        model = onnx.load('./'+ 'inceptionv3' +'.onnx')
    model, check = simplify(model)

    target_version = 13
    converted_model = version_converter.convert_version(model, target_version)
    torch_model = convert(converted_model)
    torch_model.to(args.which_device)
    torch_model.eval()
    print("[INFO] Pytorch strat infer " + "inceptionv3" + " network on " + args.which_device)
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

    print("[INFO] Magicmind strat infer inceptionv3 network on " + args.which_device)
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
