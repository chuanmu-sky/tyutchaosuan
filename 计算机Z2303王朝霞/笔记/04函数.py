# 用 def 来新建函数
def add(x, y):
    print("x is %s and y is %s" % (x, y))
    return x + y    # 通过 return 来返回值

# 调用带参数的函数
add(5, 6)  # => 输出 "x is 5 and y is 6" 返回 11

# 通过关键字赋值来调用函数
add(y=6, x=5)   # 顺序是无所谓的

# 我们也可以定义接受多个变量的函数，这些变量是按照顺序排列的
def varargs(*args):
    return args

varargs(1, 2, 3)  # => (1,2,3)


# 我们也可以定义接受多个变量的函数，这些变量是按照关键字排列的
def keyword_args(**kwargs):
    return kwargs

# 实际效果：
keyword_args(big="foot", loch="ness")  # => {"big": "foot", "loch": "ness"}

# 你也可以同时将一个函数定义成两种形式
def all_the_args(*args, **kwargs):
    print(args)
    print(kwargs)
"""
all_the_args(1, 2, a=3, b=4) prints:
    (1, 2)
    {"a": 3, "b": 4}
"""

# 当调用函数的时候，我们也可以进行相反的操作，把元组和字典展开为参数
args = (1, 2, 3, 4)
kwargs = {"a": 3, "b": 4}
all_the_args(*args)  # 等价于 for(1, 2, 3, 4)
all_the_args(**kwargs)  # 等价于 for(a=3, b=4)
all_the_args(*args, **kwargs)  # 等价于 for(1, 2, 3, 4, a=3, b=4)

# 函数在 python 中是一等公民
def create_adder(x):
    def adder(y):
        return x + y
    return adder

add_10 = create_adder(10)
add_10(3)  # => 13

# 匿名函数
(lambda x: x > 2)(3)  # => True

# 内置高阶函数
map(add_10, [1, 2, 3])  # => [11, 12, 13]
filter(lambda x: x > 5, [3, 4, 5, 6, 7])  # => [6, 7]

# 可以用列表方法来对高阶函数进行更巧妙的引用
[add_10(i) for i in [1, 2, 3]]  # => [11, 12, 13]
[x for x in [3, 4, 5, 6, 7] if x > 5]  # => [6, 7]