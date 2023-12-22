import paddle
import paddle.vision.transforms as T
from paddle.vision.datasets import Cifar10
from pyinfinitensor.onnx import OnnxStub, backend
import onnx
import itertools

def run_cifar_train_and_infer():
    
    paddle.device.set_device("gpu")

    transform = T.Compose(
        [
            T.Resize(224),
            T.ToTensor(),
            T.Normalize(
                mean=[0.5, 0.5, 0.5],
                std=[0.5, 0.5, 0.5],
                to_rgb=True,
            ),
        ]
    )
    
    # 下载数据集并初始化 DataSet
    train_dataset = paddle.vision.datasets.Cifar10(mode='train', transform=transform)
    test_dataset = paddle.vision.datasets.Cifar10(mode='test', transform=transform)

    # 模型组网并初始化网络
    inception = paddle.vision.models.InceptionV3(num_classes=10)
    model = paddle.Model(inception)

    # 模型训练的配置准备，准备损失函数，优化器和评价指标
    model.prepare(paddle.optimizer.Adam(parameters=model.parameters()), 
                paddle.nn.CrossEntropyLoss(),
                paddle.metric.Accuracy())

    # 模型训练
    model.fit(train_dataset, epochs=5, batch_size=64, verbose=1)
    # 模型评估
    model.evaluate(test_dataset, batch_size=64, verbose=1)

    # export to ONNX
    save_path = 'onnx.save/inception' # 需要保存的路径
    x_spec = paddle.static.InputSpec([1, 3, 224, 224], 'float32', 'x') # 为模型指定输入的形状和数据类型，支持持 Tensor 或 InputSpec ，InputSpec 支持动态的 shape。
    paddle.onnx.export(inception, save_path, input_spec=[x_spec], opset_version=11)

    # 加载onnx模型并放到Infinitensor中
    model_path = save_path + ".onnx"
    onnx_model = onnx.load(model_path)
    gofusion_model = OnnxStub(onnx_model, backend.cuda_runtime())
    model = gofusion_model
    model.init()

    # 启动推理
    cifar10_test = Cifar10(
        mode="test",
        transform=transform,  # apply transform to every image
        backend="cv2",  # use OpenCV as image transform backend
    )
    batch_size = 1
    total_size = 0
    total_acc = 0.0
    for data in itertools.islice(iter(cifar10_test), 10000):
        images, labels = data
        next(model.inputs.items().__iter__())[1].copyin_float(images.reshape([3*224*224]).tolist())
        model.run()
        outputs = next(model.outputs.items().__iter__())[1].copyout_float()
        outputs = paddle.to_tensor(outputs)
        outputs = paddle.reshape(outputs, (1, 10))
        labels = paddle.to_tensor(labels)
        labels = paddle.reshape(labels, (1,1))
        acc = paddle.metric.accuracy(outputs, labels)
        total_acc += acc
        total_size += batch_size
    print("test acc: {}".format(total_acc.numpy() / total_size))



if __name__ == "__main__":
    run_cifar_train_and_infer() 
