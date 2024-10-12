# 类的定义方式，使用class关键字

# 在定义类时，可以定义变量（属性）和函数（方法）

# 类属性：在类中定义的变量，是公共属性，
#     访问：类属性可以通过类或类的实例访问到（对象.属性名）
#     修改：类属性只能通过类对象来修改，无法通过实例对象修改

# 实例属性：通过实例对象添加的属性属于实例属性（在init()方法中定义）
#     实例属性只能通过实例对象来访问和修改（对象.属性名），类对象无法访问修改

# 方法：在类中定义的函数（方法），可以通过该类的所有实例来访问（对象.方法名()）
#     实例方法：在类中定义，以self为第一个参数的方法都是实例方法
#         实例方法在调用时，python会将调用对象作为self传入
#         实例方法可以通过实例和类去调用
#             当通过实例调用时，会自动将当前调用对象作为self传入
#             当通过类调用时，不会自动传递self，此时我们必须手动传递self
#     类方法：在类内部使用 @classmethod 修饰的方法
#         类方法的第一个参数是cls，也会被自动传递，cls就是当前的类对象
#             实例方法的第一个参数是self，而类方法的第一个参数是cls
#             类方法可以通过类去调用，也可以通过实例调用，没有区别
#     静态方法：在类中使用 @staticmethod 修饰的方法
#         静态方法不需要指定任何的默认参数，可以通过类和实例去调用
#         静态方法，基本上是一个和当前类无关的方法（常作为工具方法），它只是一个保存到当前类中的函数

# 我们新建的类是从 object 类中继承的
class Human(object):

     # 类属性，由所有类的对象共享
    species = "H. sapiens"

    # 基本构造函数
    def __init__(self, name):
        # 将参数赋给对象成员属性
        self.name = name

    # 成员方法，参数要有 self
    def say(self, msg):
        return "%s: %s" % (self.name, msg)

    # 类方法由所有类的对象共享
    # 这类方法在调用时，会把类本身传给第一个参数
    @classmethod
    def get_species(cls):
        return cls.species

    # 静态方法是不需要类和对象的引用就可以调用的方法
    @staticmethod
    def grunt():
        return "*grunt*"


# 实例化一个类
i = Human(name="Ian")
print(i.say("hi"))     # 输出 "Ian: hi"

j = Human("Joel")
print(j.say("hello"))  # 输出 "Joel: hello"

# 访问类的方法
i.get_species()  # => "H. sapiens"

# 改变共享属性
Human.species = "H. neanderthalensis"
i.get_species()  # => "H. neanderthalensis"
j.get_species()  # => "H. neanderthalensis"

# 访问静态变量
Human.grunt()  # => "*grunt*"