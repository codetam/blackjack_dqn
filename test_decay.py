epsilon = 1

for i in range(1, 60000):
    epsilon = epsilon * 0.99995
    if i % 5000 == 0:
        print(epsilon)