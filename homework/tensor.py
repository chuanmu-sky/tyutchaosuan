import matplotlib.pyplot as plt
import torch
import numpy as np

a = torch.tensor([1, 543, 466, 44]).reshape(2, 2)
b = torch.tensor([3, 57, 245, 2]).reshape(2, 2)

# 数组运算
c = a + b
print(c)

# 数组拼接
d = torch.cat((a, b), dim=0)
e = torch.cat((a, b), dim=1)
print(d)
print(e)

# 数组索引赋值
e[:2, 1:3] = 0
print(e)

# 取消数组重构
y = torch.zeros_like(e)
print(id(y))

y[:] = y + e
print(id(y))

# reshape造成的地址问题
x = torch.arange(1, 10)
z = x.reshape(3, 3)

print(x)
z[:] = 0
print(x)

# 按照给定维度链接张量
m = torch.tensor([1,2,3]).reshape(1,3)
n = torch.cat([m,m,m], dim=0) # 注意中括号不能省略
print(n) # 维度是几，哪个维度就会发生变化
print(torch.cat([n, n], dim=1))