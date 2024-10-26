import torch
'''----------------------------------------------------------------------------------------------'''
class Resnet:
    # 输入x, x卷积一次，激活一次, 卷积一次, 和原来的x相加, 一起激活, 两次卷积不改变长宽和通道数
    def __init__(self, channel):
        self.channel = channel
        self.c_kernel = torch.nn.Conv2d(channel, channel, kernel_size=3, padding=1)
        self.rel = torch.nn.ReLU()

    def forward(self, x_data):
        y = self.rel(self.c_kernel(x_data))
        y = self.c_kernel(y)
        y = self.rel(y + x_data)
        return y
        
    def __call__(self, x_data):
        return self.forward(x_data)
'''----------------------------------------------------------------------------------------------'''
class Model:
    # 输入(batch,1,28,28)
    # 卷积为(batch,16,24,24) 激活 池化 res 
    # 卷积为(batch,32,8,8)   激活 池化 res 
    # view为(batch,-1 ) 回归为(batch,10)
    def __init__(self): 
        self.c_kernel_1 = torch.nn.Conv2d(1, 16, kernel_size=5)
        self.c_kernel_2 = torch.nn.Conv2d(16, 32, kernel_size=5)
        
        self.rel = torch.nn.ReLU()
        
        self.pool = torch.nn.MaxPool2d(kernel_size=2)
        
        self.res_1 = Resnet(16)
        self.res_2 = Resnet(32)

        self.linear = torch.nn.Linear(512, 10)

    def forward(self, x_data):
        batch = x_data.size(0)
        
        x_data = self.pool(self.rel(self.c_kernel_1(x_data)))
        x_data = self.res_1(x_data)

        x_data = self.pool(self.rel(self.c_kernel_2(x_data)))
        x_data = self.res_2(x_data)

        x_data = x_data.view(batch, -1)
        x_data = self.linear(x_data)

        return x_data
    
    def __call__(self, x_data):
        return self.forward(x_data)

model = Model()
'''----------------------------------------------------------------------------------------------''' 
batch = 1000
x_data = torch.randn(batch, 1, 28, 28)
y = model(x_data)
print(y.size())
