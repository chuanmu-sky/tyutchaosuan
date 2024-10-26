import torch
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
'''----------------------------------------------------------------------------------------------'''
batch = 100
input_data = torch.randn(batch, 1, 28, 28)
output_data = module(input_data)
print(output_data.shape)
