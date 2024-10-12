import torch
import matplotlib.pyplot as plt
import numpy as np
# 创建初始数据集
x = torch.normal(0, 1, size=(1000, 2))  # 生成x1 x2, 都满足标准正态分布，分别是1000个
w = torch.tensor([2, -3.4]).reshape(2, 1)        # 生成w1 w2，两个权重
b = 4.2
y = torch.mm(x, w) + b


# 机器学习
w_s = torch.normal(0, 0.01, size=(2, 1), requires_grad=True)  # 创建机器学习的w与b
b_s = torch.zeros(1, requires_grad=True)

def y_s(x, w, b):
    return torch.mm(x, w) + b
def loss(y, yt):
    return  (y - yt)**2 /2    # 创建损失函数

for i in range(150):
    temp = y_s(x, w_s, b_s)
    l = loss(y, temp)  
    l.sum().backward()

    
    with torch.no_grad():
        w_s -= w_s.grad * (0.04/1000)      # ls为0.04，但是梯度是求和后的梯度，梯度值扩大1000倍
        b_s -= b_s.grad * (0.04/1000)
        w_s.grad.zero_()
        b_s.grad.zero_()


# 结果验证
print(w_s)
print(b_s)