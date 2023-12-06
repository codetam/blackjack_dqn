eps = 1
decay = 0.999977
for i in range(1, 100001):
    eps *= decay
    if i % 10000 == 0:
        print(i, eps)

