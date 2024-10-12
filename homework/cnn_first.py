import torch
'''----------------------------------------------------------------------------------------------'''
class Cnn_Model:
    # 把 (batch,1,28,28) 转变成 (batch,10)
    def __init__(self):
        # 两次卷积
        self.cnn_1 = torch.nn.Conv2d(1, 10, kernel_size=5) # 28变24
        self.cnn_2 = torch.nn.Conv2d(10, 20, kernel_size=5) # 12变8
        # 两次激活
        self.rel = torch.nn.ReLU() # 激活函数
        # 两次池化
        self.pool = torch.nn.MaxPool2d(kernel_size=2) # 24变12 8变4
        # 三次线性回归
        self.linear_1 = torch.nn.Linear(320, 160)
        self.linear_2 = torch.nn.Linear(160, 80)
        self.linear_3 = torch.nn.Linear(80, 10)

    def forward(self, x):
        batch_size = x.size(0)
        # 两次卷积+池化+激活
        x = self.rel(self.pool(self.cnn_1(x)))
        x = self.rel(self.pool(self.cnn_2(x)))
        # 三次线性回归
        x = x.view(batch_size, -1)
        x = self.linear_1(x)
        x = self.linear_2(x)
        x = self.linear_3(x)

        return x 

    def __call__(self, x):
        return self.forward(x)

model = Cnn_Model()
'''----------------------------------------------------------------------------------------------'''
batch = 100
input_data = torch.randn(batch, 1, 28, 28)
output_data = model(input_data)
print(output_data.shape)