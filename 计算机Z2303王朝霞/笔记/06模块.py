# python模块（module）
# 是一个python文件,以.py结尾.包含python对象定义和python语句
# 模块能定义函数、类和变量，模块里也能包含可执行的代码

# 我们可以导入其他模块
import math
print (math.sqrt(16))  # => 4

# 我们也可以从一个模块中导入特定的函数
from math import ceil, floor
print (ceil(3.7))   # => 4.0
print (floor(3.7))  # => 3.0

# 从模块中导入所有的函数
# 警告：不推荐使用
from math import *

# 简写模块名
import math as m
math.sqrt(16) == m.sqrt(16)  # => True

# Python的模块其实只是普通的python文件
# 你也可以创建自己的模块，并且导入它们
# 模块的名字就和文件的名字相同

# 也可以通过下面的方法查看模块中有什么属性和方法
import math
dir(math)


# 示例 time模块

# time.time(): 获取当前时间，返回的是一个数字
# time.sleep(秒数): 休眠

import time
star = time.time()  # 程序开始的时间
list = []
for i in range(100000):
    list.append(i)
end = time.time()  # 程序结束的时间

print(f'程序运行的时间为{star - end}s时间')

# 生成随机数
import random
n=random.randiant(1,100) # 生成1-100（包含1和100）的随机整数
p=random.randrange(1,100,2) # 生成1-100（不包含100）的随机整数，2为步长
m=random.uniform(1,100) # 生成1-100的随机小数
t=random.random() # 生成0-1的小数


# 自定义模块
# 模块可以包含：全局变量、函数和类

# 创建一个自定义文件，如my_module1

def sum_num(num1, num2):
    return num1 + num2

# 在其他文件中导入自定义模块并调用自定义模块中的函数
"""
import my_module1
my_module1.sum_num(10, 20)

"""