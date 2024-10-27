def max_sum(a):
    max_s = a[0]
    now_s = a[0]

    for i in a[1:]:
        now_s = max(i, now_s + i)
        max_s = max(max_s, now_s)
    if max_s<0:
        max_s=0
    return max_s

a = [-20,11,-4,13,-5,-2]
print(max_sum(a))