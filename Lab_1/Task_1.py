def my_or(x1, x2):
    return x1 or x2

def my_and(x1, x2):
    return x1 and x2

# Реалізація XOR через OR та AND
# Відомо, що: XOR = (x1 OR x2) AND NOT(x1 AND x2)
def my_xor(x1, x2):
    return my_and(my_or(x1, x2), not my_and(x1, x2))

# Тестування
print("x1 x2 | XOR")
for x1 in [0, 1]:
    for x2 in [0, 1]:
        print(f" {x1}  {x2} | {int(my_xor(x1, x2))}")
