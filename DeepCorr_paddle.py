# image 处理为三维[通道数，宽，高]
# Conv2D 接收四维数据[batch, 通道数， 宽， 高]
import paddle
import paddle.nn.functional as F
import h5py
import paddle.nn as nn

class MyDataset(paddle.io.Dataset):
    def __init__(self):
        super(MyDataset, self).__init__()
        self.data_list = []
        with h5py.File('data/tordata300.h5', 'r') as f:
            x = list(f['data']['x'])
            y = list(f['data']['y'])
        for i in range(0, len(x)):
            self.data_list.append([x[i], y[i]])

    def __getitem__(self, index):
        """
        实现 __getitem__ 函数，定义指定 index 时如何获取数据，并返回单条数据（样本数据、对应的标签）
        """
        # 根据索引，从列表中取出一个图像
        image, label = self.data_list[index]
        # CrossEntropyLoss要求label格式为int，将Label格式转换为 int
        image = paddle.to_tensor(image)
        #image = paddle.reshape(image, [1, 8, 300])
        label = int(label)
        # 返回图像和对应标签
        return image, label

    def __len__(self):
        """
        实现 __len__ 函数，返回数据集的样本总数
        """
        return len(self.data_list)


'''class MNIST(paddle.nn.Layer):
    def __init__(self):
        super(MNIST, self).__init__()
        self.conv1 = paddle.nn.Conv2D(in_channels=1, out_channels=8, kernel_size=(2, 2))
        self.pool1 = paddle.nn.MaxPool2D(kernel_size=2, stride=2)
        self.F1 = paddle.nn.Flatten(1, -1)
        self.Cov = paddle.nn.Conv2D
        # 定义一层全连接层，输出维度是1
        self.fc = paddle.nn.Linear(in_features=3576, out_features=1)
    # 定义网络结构的前向计算过程
    def forward(self, inputs):
        x = self.conv1(inputs)
        x = F.relu(x)
        x = self.pool1(x)
        x = self.F1(x)
        outputs = self.fc(x)
        return outputs'''
model = paddle.nn.Sequential(
    nn.Conv2D(1, 8, 2, stride=1, padding=1),
    nn.ReLU(),
    nn.MaxPool2D(2, 2),
    nn.Flatten(),
    nn.Linear(4800, 1),
)

train_dataset = MyDataset()


def norm_img(img):
    # 验证传入数据格式是否正确，img的shape为[batch_size, 28, 28]
    assert len(img.shape) == 3
    batch_size, img_h, img_w = img.shape[0], img.shape[1], img.shape[2]
    # 归一化图像数据
    img = img / 255
    img = paddle.reshape(img, [batch_size,1, 8, 300])
    return img


# # 声明网络结构
# model = MNIST()


def train(model):
    # 启动训练模式
    model.train()
    # 定义并初始化数据读取器
    train_loader = paddle.io.DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=1,
                                        drop_last=True)

    # 定义优化器，使用随机梯度下降SGD优化器，学习率设置为0.001
    opt = paddle.optimizer.SGD(learning_rate=0.001, parameters=model.parameters())
    EPOCH_NUM = 10
    for epoch in range(EPOCH_NUM):
        for batch_id, data in enumerate(train_loader()):
            images = norm_img(data[0]).astype('float32')
            labels = data[1].astype('float32')
            # 前向计算的过程
            predicts = model(images)
            # 计算损失
            loss = F.square_error_cost(predicts, labels)
            avg_loss = paddle.mean(loss)

            # 每训练了1000批次的数据，打印下当前Loss的情况
            if batch_id % 1000 == 0:
                print("epoch_id: {}, batch_id: {}, loss is: {}".format(epoch, batch_id, avg_loss.numpy()))

            # 后向传播，更新参数的过程
            avg_loss.backward()
            opt.step()
            opt.clear_grad()


train(model)
paddle.save(model.state_dict(), './mnist.pdparams')
