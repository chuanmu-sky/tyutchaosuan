import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import numpy as np
import math
'''----------------------------------------------------------------------------------------------'''
class Getdata(Dataset):
    # 得到数据，得到第loc个x与y，得到n的值
    def __init__(self, filepath):
        super(Getdata,self).__init__()
        self.xy = np.loadtxt(filepath, delimiter=',', dtype=np.float32) # 文件路径，字符分割符号，字符数据类型
        self.len = self.xy.shape[0] # 每组数据都是8个x与一个y输出，一共有n组，shape=(n,9)
        self.x = torch.from_numpy(self.xy[:, :-1])
        self.y = torch.from_numpy(self.xy[:, -1])  # 把x和y的数据提出

    def __getitem__(self, loc):
        return self.x[loc], self.y[loc]  # 魔法方法，定义利用loc参数寻找第loc个x与y的值 

    def __len__(self):
        return self.len # 魔法方法，返回n的值

my_data = Getdata(r"C:\Users\61585\Desktop\diabetes.csv.gz")
train_data = DataLoader(dataset=my_data, batch_size=32, shuffle=True) # 必须是定义了getitem和len的方法的对象，设置每个批量，设置打乱
'''----------------------------------------------------------------------------------------------'''
class Model(torch.nn.Module):
    # 得到y_hat，同时利用Linear得到线性回归方程，Linear会自动创建w与b
    def __init__(self):
        super(Model, self).__init__()  # 初始化父类
        # 定义四个
        self.get_linear_1 = torch.nn.Linear(8, 6)
        self.get_linear_2 = torch.nn.Linear(6, 4)
        self.get_linear_3 = torch.nn.Linear(4, 1) # linear根据x和y的列数确定w的大小，并加入偏置
        self.get_sigmoid  = torch.nn.Sigmoid()

    def forward(self, x_data): # 对父类函数的重载
        x_data = self.get_sigmoid(self.get_linear_1(x_data)) # 均是用了call的魔法方法
        x_data = self.get_sigmoid(self.get_linear_2(x_data))
        x_data = self.get_sigmoid(self.get_linear_3(x_data))
        return x_data # 构建多层神经网络并对每一层使用sig函数激活
    
model = Model()
'''----------------------------------------------------------------------------------------------'''
# criterion = torch.nn.BCELoss(reduction='mean')  维度问题无法解决
optimizer = torch.optim.SGD(model.parameters(), lr=0.01) # 求优化，第一个参数是优化对象，第二个参数是学习率

def myloss(y_hat,y):
    sum = 0
    for i in range(len(y)):
        sum+=-(y_hat[i]*math.log(y[i]+0.00000000001)+(1-y_hat[i])*math.log(1-y[i]+0.0000000001))
        return sum/len(y)
'''----------------------------------------------------------------------------------------------'''
for epoch in range(1000):
    for i ,data in enumerate(train_data, 0): # enumerate的参数：一个可以迭代的对象，开始的索引值。注意这里的索引值是指返回给i的，并不是从train_data的第0个开始迭代
        x, y = data # 传入数据
        y_hat = model.forward(x) # 求y_hat
        # loss = criterion(y_hat,y) # 求损失函数，但是一会求导时需要标量，所以将其标量化
        loss = myloss(y_hat,y)
        print(loss.item())
        print(model.parameters())
        optimizer.zero_grad() # 梯度清零
        loss.backward() # 求梯度
        optimizer.step() # 梯度更新
        
