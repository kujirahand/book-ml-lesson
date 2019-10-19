import random

data_count = 100000

fat = 0
lean = 0
normal = 0 
res = ""
for i in range(data_count):
    cm = random.randrange(1500, 1850) / 10
    kg = random.randrange(350, 890) / 10
    bmi = kg / ((cm / 100) ** 2)
    if bmi < 18.5:
        lean += 1
        val = '痩せ'
    elif bmi < 25.0:
        fat += 1
        val = '普通'
    else:
        normal += 1
        val = '肥満'
    res += "{},{},{}\n".format(kg, cm, val)

fp = open("bodydata.csv", "wt")
fp.write(res)
fp.close()

print(res)
print(lean, normal, fat)

