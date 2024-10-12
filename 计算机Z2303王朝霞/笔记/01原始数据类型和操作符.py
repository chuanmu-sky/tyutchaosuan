# 数字类型
3  # => 3

# 简单的算数
1 + 1  # => 2
8 - 1  # => 7
10 * 2  # => 20
35 / 5  # => 7

# 整数的除法会自动取整
5 / 2  # => 2

# 要做精确的除法，我们需要引入浮点数
2.0     # 浮点数
11.0 / 4.0  # => 2.75 精确多了

# 括号具有最高优先级
(1 + 3) * 2  # => 8

# 布尔值也是基本的数据类型
True
False

# 用 not 来取非
not True  # => False
not False  # => True

# bool类型有三种类型运算，分别是与（and） 或（or） 非（not）
d=False
e=not d
f=((not e or d) and e) or d

# 相等
1 == 1  # => True
2 == 1  # => False

# 不等
1 != 1  # => False
2 != 1  # => True

# 更多的比较操作符
1 < 10  # => True
1 > 10  # => False
2 <= 2  # => True
2 >= 2  # => True

# 比较运算可以连起来写！
1 < 2 < 3  # => True
2 < 3 < 2  # => False

# 字符串通过 " 或 ' 括起来
"This is a string."
'This is also a string.'

# 字符串通过加号拼接
"Hello " + "world!"  # => "Hello world!"

# 字符串可以被视为字符的列表
"This is a string"[0]  # => 'T'

# % 可以用来格式化字符串
"%s can be %s" % ("strings", "interpolated")

# 也可以用 format 方法来格式化字符串
# 推荐使用这个方法
s = "{0} is a {1}".format('Tom', 'Boy')
print(s)             # Tom is a Boy
# 也可以用变量名代替数字
"{name} wants to eat {food}".format(name="Bob", food="lasagna")

# 字符串用+拼接
str1='ni'
str2='hao'
print(len(str1))
str3=str1+str2    # nihao
# 字符串可以用*复制
str4=str1*2 # nini
str5=str1+str2*2
# 字符串的切片
str_cut=str3[0:5] # 把字符串切取出来一段，字母从0开始，本例提取字符串中第0-4个元素，不包括左边界
                  # 如果不写结束会议直接取到结尾的元素，如果开始和结束都没写会截取全部
                  # 可以加第三个参数，表示步长
print(str_cut)
str_=str3[-5:-1] # 从倒数第五个到倒数第一个（不包括倒数第一个）
print(str_)
str_re=str3[::-1] # 表示把字符串反过来
str3=str3.replace("ni","wo") # 替换字符串里的某一段内容
print(str3)
str6='my name is xxx'
str6=str6.split(' ') # 将字符串以空格分割，得到一个列表
print(str6)

# None 是对象
None  # => None

# 不要用相等 `==` 符号来和None进行比较
# 要用 `is`
"etc" is None  # => False
None is None  # => True

# 'is' 可以用来比较对象的相等性
# 这个操作符在比较原始数据时没多少用，但是比较对象时必不可少

# None, 0, 和空字符串都被算作 False
# 其他的均为 True
0 == False  # => True
"" == False  # => True