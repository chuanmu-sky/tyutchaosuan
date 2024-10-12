import torch

# 自动求导：y为标量时才能求导
x_1 = torch.arange(4.0)
x_1.requires_grad_(True)   # 创建tensor后使用是两个下划线，并且是作为一个mean出现，true说明这个tensor需要求导
y_1 = torch.dot(x_1, x_1)
y_1.backward()             # 求导
print(x_1.grad)

x_2 = torch.arange(0.0, 4.0, requires_grad=True)   # 创建tensor时使用是一个下划线，并且不是mean，同上的true
y_2 = torch.dot(x_2, x_2)
y_2.backward(retain_graph=True)  # 因为要求二阶导，所以要对n的一阶导的函数进行保留(即保留计算图，默认会销毁)
print(x_2.grad)
y_2.backward()
print(x_2.grad)

x_1.grad.zero_()           # y与x的函数关系变化时，再求导数要将之前的梯度清零
y_3 = x_1.sum()
y_3.backward()
print(x_1.grad)



# 自动求导：y为非标量时要转化成标量再求导
x = torch.arange(4.0, requires_grad=True)
y = x * x
y.sum().backward()  # 对y求和后的标量对x的导数，和y直接对x的导数是一样的
print(x.grad)

x.grad.zero_()
yy = x * x
yy.backward(torch.ones_like(yy))
print(x.grad)

