import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
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
def cnn_block(input_channel, output_channel):
    # 每个dense都是由n个卷积块组成的，所以先封装一个卷积块方便定义
    return torch.nn.Sequential(
        torch.nn.BatchNorm2d(input_channel),  # 输入的数据是(batch,channel,height,width), 这里的参数规定channel的大小, 电脑自动bn化 
        torch.nn.ReLU(),
        torch.nn.Conv2d(input_channel, output_channel, kernel_size=3, padding=1)
    )

class Dense_Block(torch.nn.Module):
    # dense的组成，n个卷积块，注意每次卷积的时候输入与输出的通道数，注意每个输入层的拼接
    def __init__(self, n, input_channel, output_channel):
        super(Dense_Block, self).__init__()
        layer = []
        for i in range(n):
            # 第一次 输入a通道, 输出b通道
            # 第二次 a+b ~ b
            # 第三次 a+2b ~ b
            # 第n次  a+(n-1)b ~ b 即a+ib ~ b
            layer.append(cnn_block(output_channel*i+input_channel, output_channel))
        self.net = torch.nn.Sequential(*layer)

    def forward(self, x):
        # 完成每一层的卷积和每一层输入的拼接
        for i in self.net:
            y = i(x)                     # 魔法方法, 这里完成的是本层的卷积
            x = torch.cat((x, y), dim=1) # 这里完成的是下一层的输入拼接

        return x
'''----------------------------------------------------------------------------------------------'''
def transition(input_channel, output_channel):
    # 每次使用dense后都会导致通道数增加，在这里降低通道数
    return torch.nn.Sequential(
        torch.nn.BatchNorm2d(input_channel),
        torch.nn.ReLU(),
        torch.nn.Conv2d(input_channel, output_channel, kernel_size=1), # 注意这里卷积核的大小是1
        torch.nn.AvgPool2d(kernel_size=2, stride=2)
    )
'''----------------------------------------------------------------------------------------------'''
# 定义一个模型
def Module():
    # (batch,1,28,28) ~ Cnn(batch,8,24,24) ~ BN ~ Relu ~ MaxPool(batch,8,12,12)
    b1 = torch.nn.Sequential(
        torch.nn.Conv2d(1, 8, kernel_size=5),
        torch.nn.BatchNorm2d(8),
        torch.nn.ReLU(),
        torch.nn.MaxPool2d(kernel_size=2, stride=2)
    )
    # 四个dense, 每个dense有4次卷积, 增长率为16, 且每个dense中间安排一个transition, 一共安排3个
    layer = []
    input_channel = 8
    for i in range(4):                                   # 4个dense
        layer.append(Dense_Block(4, input_channel, 16))  # 第i个dense
        input_channel += 16*4                            # 经过第i个dense后输入通道的改变
        if i <= 2:
            layer.append(transition(input_channel, input_channel//2)) # 降低标准为减半
            input_channel = input_channel // 2

    return torch.nn.Sequential(
        b1, *layer,
        torch.nn.BatchNorm2d(input_channel),
        torch.nn.ReLU(),
        torch.nn.Flatten(), # 展开函数，(batch,channel,height,weight) ~ (batch,c*h*w)
        torch.nn.Linear(input_channel, 10)  # ????????????????
    )

module = Module()
module.to(device)
'''----------------------------------------------------------------------------------------------'''
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(module.parameters(), lr=0.1)
'''----------------------------------------------------------------------------------------------'''
for epoch in range(30):
    # 训练一次
    for i,data in enumerate(mnist_train, 0):
        inputs, targets = data
        inputs = inputs.to(device)
        targets = targets.to(device)
        outputs = module(inputs)
    
        optimizer.zero_grad()
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

    print(loss.item())
    
    # 测试一次
    correct = 0
    for inputs,targets in mnist_test:
        inputs = inputs.to(device)
        targets = targets.to(device)
        outputs = module(inputs)
        prediction = torch.argmax(outputs, dim=1)    # 返回最大值的索引值，在行上寻找10个里面的最大的
        correct += torch.sum(prediction == targets)  # 索引值是0~9，对应的数字恰好也是0~9

    result = correct/(256 * len(mnist_test))
    print(result)
'''----------------------------------------------------------------------------------------------'''
