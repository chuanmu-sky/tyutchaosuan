# ccf-csp练习

# 201312-1 出现次数最多的数
# n=int(input())
# a=list(map(int,input().split()))
# a.sort()
# num=1
# max=1
# index=0
# for i in range(n-1):
#     if a[i]==a[i+1]:
#         num+=1
#     else:
#         if num>max:
#             max=num
#             index=i
#         num=1
# print(a[index])

# 202312-1 仓库规划
# n, m = map(int, input().split())
# a=[]
# for i in range(n):
#      a.append([])
#      a[i]=map(int,input().split())
# for i in range(n):
#     output = 0
#     for j in range(n):
#         if i!=j:
#             flag = True
#             for k in range(m):
#                 if a[i][k]>=a[j][k]:
#                     flag = False
#                     break
#             if flag:
#                 output = j+1
#                 break
#     print(output)

# 202309-1 坐标变换
# n,m=map(int,input().split())
# delta_x=0
# delta_y=0
# for i in range(n):
# 	dx,dy=map(int,input().split())
# 	delta_x+=dx
# 	delta_y+=dy
# a=[]
# for i in range(m):
# 	a.append([])
# 	a[i]=list(map(int,input().split()))
# for i in range(m):
# 	a[i][0]+=delta_x
# 	a[i][1]+=delta_y
# 	print("{0} {1}".format(a[i][0],a[i][1]))

# 202305-1 重复局面
# n = int(input())
# chess = {}
# for i in range(n):
#     temp = ''
#     for j in range(8):
#         temp += input()
#     if temp not in chess:
#         chess[temp] = 1
#     else:
#         chess[temp] += 1
#     print(chess[temp])

# 202303-1 田地丈量
# n, a, b = map(int, input().split())
# sum = 0
# for i in range(n):
#     x1,y1,x2,y2=map(int,input().split())
#     x = min(a, x2)-max(0, x1)
#     y = min(b, y2)-max(0, y1)
#     if x>=0 and y>=0:
#         sum += x*y
# print(sum)

# 202212-1 现值计算
# n,i=map(float,input().split())
# n=int(n)
# a=list(map(int,input().split()))
# sum=0
# for j in range(n+1):
# 	sum+=a[j]*pow(1+i,-j)
# print(sum)

# 202206-1 归一化处理
# n = int(input())
# nums = list(map(int,input().split()))
# a = sum(nums)/n
# d = 0
# for i in range(n):
#     d += (nums[i]-a)**2
# d /= n
# data = []
# for i in range(n):
#     point = (nums[i]-a)/(d**0.5)
#     data.append(point)
# for i in data:
#     print(i)

# 202203-1 未初始化警告
n,k=map(int,input().split())
a=set(0)
count=0
for i in range(k):
    x,y=map(int,input().split())
    if y not in a:
        count+=1
    a.add(x)
    if len(a)>n+1:
        break
print(count)