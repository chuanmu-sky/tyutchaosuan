import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
import numpy as np
'''----------------------------------------------------------------------------------------------'''
def getitem():
    # 加载数据集，把数据集中的训练集和测试集以minibatch形式返回
    train = torchvision.datasets.MNIST(
        root="D:\\develop\\data_train\\mnist",        # 加载数据集的地址
        transform=ToTensor(),                         # 转变为张量
        train=True,                                   # 加载数据的类型
        download=False)         # 下载数据集，第一次需要联网：download=true，下载好以后不再需要联网：download=false
    test = torchvision.datasets.MNIST(
        root="D:\\develop\\data_train\\mnist",
        transform=ToTensor(),
        train=False,
        download=False)
    '''
    print(len(mnist_test))         # 一共10000份数据
    print(mnist_test[0][0].shape)  # 第0份数据的第0个位置是一个图片
    print(mnist_test[0][1])        # 第0份数据的第1个位置是对应的数字
    '''
    return DataLoader(dataset=train, batch_size=256, shuffle=True), DataLoader(dataset=test, batch_size=256, shuffle=True)


mnist_train, mnist_test = getitem()
'''----------------------------------------------------------------------------------------------'''
class Resnet:
    # 输入x, 卷积一次, 激活一次, 卷积一次, 和原来的x相加, 一起激活, 两次卷积不改变大小
    def __init__(self,channel):
        self.c_kernel = torch.nn.Conv2d(channel, channel, kernel_size=3, padding=1)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        y = self.relu(self.c_kernel(x))
        y = self.relu(self.c_kernel(y) + x)
        return y
    
    def __call__(self, x):
        return self.forward(x)
    

class Model(torch.nn.Module):
    # 卷积 (batch,1,28,28) 变为 (batch,10,24,24)
    # 池化 (batch,10,24,24) 变为 (batch,10,12,12)
    # resnet
    # 卷积 (batch,10,12,12) 变为 (batch,20,8,8)
    # 池化 (batch,20,8,8) 变为 (batch,20,4,4)
    # resnet
    # 展平 + linear为 (batch,10)
    # softmax
    def __init__(self): 
        super(Model, self).__init__()
        self.c_kernel_1 = torch.nn.Conv2d(1, 10, kernel_size=5)
        self.c_kernel_2 = torch.nn.Conv2d(10, 20, kernel_size=5)
        self.m_pool = torch.nn.MaxPool2d(kernel_size=2)
        self.res_1 = Resnet(10)
        self.res_2 = Resnet(20)
        self.linear_1 = torch.nn.Linear(320, 50)
        self.linear_2 = torch.nn.Linear(50, 10)

    def forward(self, x):
        x = self.res_1(self.m_pool(self.c_kernel_1(x)))
        x = self.res_2(self.m_pool(self.c_kernel_2(x)))
        x = x.view(-1, 320)
        x = self.linear_2(self.linear_1(x))
        x = torch.nn.functional.softmax(x, dim=1)
        return x
    

model = Model()
'''----------------------------------------------------------------------------------------------'''
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.03)
'''----------------------------------------------------------------------------------------------'''
for epoch in range(30):
    # 训练一次
    for i,data in enumerate(mnist_train, 0):
        inputs, targets = data
        outputs = model(inputs)
    
        optimizer.zero_grad()
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        print(loss.item())
    
    # 测试一次
    correct = 0
    for inputs,targets in mnist_test:
        outputs = model(inputs)
        prediction = torch.argmax(outputs, dim=1)    # 返回最大值的索引值，在行上寻找10个里面的最大的
        correct += torch.sum(prediction == targets)  # 索引值是0~9，对应的数字恰好也是0~9

    result = correct/(256 * len(mnist_test))
    print(result)
'''----------------------------------------------------------------------------------------------'''