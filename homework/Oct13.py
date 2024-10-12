# 分治策略
def getsum(list, left, right):
    # 处理只有一个数据的数组
    if left == right:
        return max(list[left], 0)
        
    # 处理有多个数据的数组
    else:
        # 递归调用        
        center = int((left + right) / 2)
        leftsum = getsum(list, left, center)
        rightsum = getsum(list, center + 1, right)
        
        # 对最大和包含center的情况处理
        lefta = 0
        righta = 0
        temp = [0, 0]

        for i in range(center, left - 1 , -1):
            lefta += list[i]
            temp[0] = max(temp[0], lefta)
        
        for i in range(center + 1, right + 1):
            righta += list[i]
            temp[1] = max(temp[1], righta)
        
        return max(leftsum, rightsum, sum(temp))
    
x = [-2, 11, -4, 13, -5, -2, 8]
print(x)
print(getsum(x, 0, len(x)-1))  

# 动态规划
def getsum(list):
    sum = 0
    a = 0
    # 固定左端
    for i in range(0, len(list)):
        if a > 0:
            a += list[i]
        else:
            a = list[i]
        sum = max(a, sum)

    return sum

x = [-2, 11, -4, 13, -5, -2, 5]
print(x)
print(getsum(x))  