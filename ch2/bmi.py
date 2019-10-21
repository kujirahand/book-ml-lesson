import sys

# コマンドライン引数を処理
if len(sys.argv) < 3:
    print("bmi.py kg cm")
    quit()
kg = float(sys.argv[1])
cm = float(sys.argv[2])

# BMIの計算式 --- (*1)
bmi = kg / ((cm / 100) ** 2)

# 肥満判定を行う
if bmi < 18.5:
    print("痩せ")
elif bmi > 25:
    print("肥満")
else:
    print("普通")
    
