import sys

if len(sys.argv) < 3:
    print("bmi.py kg cm")
    quit()
kg = float(sys.argv[1])
cm = float(sys.argv[2])

bmi = kg / ((cm / 100) ** 2)
if bmi < 18.5:
    print("痩せ")
elif bmi > 25:
    print("肥満")
else:
    print("普通")
    
