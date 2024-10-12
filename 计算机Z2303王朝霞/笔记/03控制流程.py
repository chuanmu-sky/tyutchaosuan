# 新建一个变量
some_var = 5

# 这是个 if 语句，在 python 中缩进是很重要的。
# 下面的代码片段将会输出 "some var is smaller than 10"
if some_var > 10:
    print("some_var is totally bigger than 10.")
elif some_var < 10:    # 这个 elif 语句是不必须的
    print("some_var is smaller than 10.")
else:           # 这个 else 也不是必须的
    print("some_var is indeed 10.")


# 用for循环遍历列表
# 输出:
#     dog is a mammal
#     cat is a mammal
#     mouse is a mammal

for animal in ["dog", "cat", "mouse"]:
    # 可以用 % 来格式化字符串
    print("%s is a mammal" % animal)

# `range(number)` 返回从0到给定数字的列表
# 输出:
#     0
#     1
#     2
#     3

for i in range(4):
    print(i)

# while 循环
# 输出:
#     0
#     1
#     2
#     3

x = 0
while x < 4:
    print(x)
    x += 1  #  x = x + 1 的简写

# 可以用列表、字符串、元组、集合自定义循环变量
arr=[1,3,5,7,9]
for i in arr:
    print(i,end='')
str='ABCDEF'
for i in str:
    print(i,end='')
    if i=='D':
        break
else:
    print("循环正常结束") # 跟在循环后面的else在循环正常循环结束后执行
                         # 若循环在中途中退出则不执行该语句

# 用 try/except 块来处理异常

# Python 2.6 及以上适用:
try:
    # 用 raise 来抛出异常
    raise IndexError("This is an index error")
except IndexError as e:
    pass    # pass 就是什么都不做，不过通常这里会做一些恢复工作