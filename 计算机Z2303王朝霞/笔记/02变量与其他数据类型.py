# 很方便的输出
print("I'm Python. Nice to meet you!")

# 给变量赋值前不需要事先声明
some_var = 5    # 一般建议使用小写字母和下划线组合来做为变量名
some_var  # => 5

# 访问未赋值的变量会抛出异常
# 可以查看控制流程一节来了解如何异常处理
# some_other_var  # 抛出 NameError

# if 语句可以作为表达式来使用
"yahoo!" if 3 > 2 else 2  # => "yahoo!"

# 列表用来保存序列
li = []
# 可以直接初始化列表
other_li = [4, 5, 6]

# 在列表末尾添加元素
li.append(1)    # li 现在是 [1]
li.append(2)    # li 现在是 [1, 2]
li.append(4)    # li 现在是 [1, 2, 4]
li.append(3)    # li 现在是 [1, 2, 4, 3]
# 移除列表末尾元素
li.pop()        # => 3 li 现在是 [1, 2, 4]
# 重新加进去
li.append(3)    # li is now [1, 2, 4, 3] again.
# 排序
li.sorted()     # li现在是[1，2，3，4]   

# 像其他语言访问数组一样访问列表
li[0]  # => 1
# 访问最后一个元素
li[-1]  # => 3

# 越界会抛出异常
li[4]  # 抛出越界异常

# 切片语法需要用到列表的索引访问
# 可以看做数学之中左闭右开区间
li[1:3]  # => [2, 4]
# 省略开头的元素
li[2:]  # => [4, 3]
# 省略末尾的元素
li[:3]  # => [1, 2, 4]

# 删除特定元素
del li[2]  # li 现在是 [1, 2, 3]

# 合并列表
li + other_li  # => [1, 2, 3, 4, 5, 6] - 并不会不改变这两个列表

# 通过拼接来合并列表
li.extend(other_li)  # li 是 [1, 2, 3, 4, 5, 6]

# 用 in 来返回元素是否在列表中
1 in li  # => True

# 返回列表长度
len(li)  # => 6


# 元组类似于列表，但它是不可改变的
tup = (1, 2, 3)
tup[0]  # => 1
tup[0] = 3  # 类型错误

# 对于大多数的列表操作，也适用于元组
len(tup)  # => 3
tup + (4, 5, 6)  # => (1, 2, 3, 4, 5, 6)
tup[:2]  # => (1, 2)
2 in tup  # => True

# 可以将元组解包赋给多个变量
a, b, c = (1, 2, 3)     # a 是 1，b 是 2，c 是 3
# 如果不加括号，将会被自动视为元组
d, e, f = 4, 5, 6
# 现在我们可以看看交换两个数字是多么容易的事
e, d = d, e     # d 是 5，e 是 4


# 字典用来储存映射关系
empty_dict = {}
# 字典初始化
filled_dict = {"one": 1, "two": 2, "three": 3}

# 字典也用中括号访问元素
filled_dict["one"]  # => 1

# 把所有的键保存在列表中
filled_dict.keys()  # => ["three", "two", "one"]
# 键的顺序并不是唯一的，得到的不一定是这个顺序

# 把所有的值保存在列表中
filled_dict.values()  # => [3, 2, 1]
# 和键的顺序相同

# 判断一个键是否存在
"one" in filled_dict  # => True
1 in filled_dict  # => False

# 查询一个不存在的键会抛出 KeyError
filled_dict["four"]  # KeyError

# 用 get 方法来避免 KeyError
filled_dict.get("one")  # => 1
filled_dict.get("four")  # => None
# get 方法支持在不存在的时候返回一个默认值
filled_dict.get("one", 4)  # => 1
filled_dict.get("four", 4)  # => 4

# setdefault 是一个更安全的添加字典元素的方法
filled_dict.setdefault("five", 5)  # filled_dict["five"] 的值为 5
filled_dict.setdefault("five", 6)  # filled_dict["five"] 的值仍然是 5


# 集合储存无顺序的元素
empty_set = set()
# 初始化一个集合
some_set = set([1, 2, 2, 3, 4])  # some_set 现在是 set([1, 2, 3, 4])

# Python 2.7 之后，大括号可以用来表示集合
filled_set = {1, 2, 2, 3, 4}  # => {1 2 3 4}

# 向集合添加元素
filled_set.add(5)  # filled_set 现在是 {1, 2, 3, 4, 5}

# 用 & 来计算集合的交
other_set = {3, 4, 5, 6}
filled_set & other_set  # => {3, 4, 5}

# 用 | 来计算集合的并
filled_set | other_set  # => {1, 2, 3, 4, 5, 6}

# 用 - 来计算集合的差
{1, 2, 3, 4} - {2, 3, 5}  # => {1, 4}

# 用 in 来判断元素是否存在于集合中
2 in filled_set  # => True
10 in filled_set  # => False