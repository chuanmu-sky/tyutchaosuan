import torch

# 矩阵的转置
a = torch.arange(1, 10).reshape(3, 3)
print(a.T)

# mm乘法：两个二维矩阵的数学乘法
b = torch.arange(10).reshape(2, 5)
c = torch.ones_like(b).reshape(5, 2)
print(torch.mm(b, c))

# mv乘法：一个二维矩阵和一个列向量的数学乘法
o = torch.arange(0, 5)
print(torch.mv(b, o))

# 矩阵的对应元素的乘法
d = torch.arange(1, 10).reshape(3, 3)
print(a*d)

# dot乘法：两个列向量的乘法
e = torch.arange(1, 10)
f = torch.arange(1, 10)
print(torch.dot(e, f))

# 矩阵的求和降维
x = torch.tensor([[1, 2, 3],
                  [4, 5, 6]])
print(x.sum(axis=0))        # 沿着哪个求和，哪个就消失，shape为(2，3)，沿着0求和，2消失，结果为横着3
print(x.sum(axis=1, keepdim=True))  # 沿着1求和，3消失，结果为竖着2，只有keepdim后才能继续参与原维度运算，否则降维后无法运算

# 矩阵的范数：矩阵的模(注意必须是浮点数)
p = torch.arange(1.0, 5.0).reshape(2, 2)
print(torch.norm(p))
