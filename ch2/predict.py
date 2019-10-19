from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import sys, pickle

# モデルを読み込み
cls = pickle.load(open("bodydata_model.sav", 'rb'))

if len(sys.argv) < 3:
    print("predict.py kg cm_")
    quit()
kg = float(sys.argv[1])
cm = float(sys.argv[2])

# 予測
y_pred = cls.predict([[kg, cm]])
print("predict=", y_pred)

