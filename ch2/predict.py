from sklearn import svm
import sys, pickle

# コマンドライン引数を確認
if len(sys.argv) < 3:
    print("predict.py kg cm_")
    quit()
kg = float(sys.argv[1])
cm = float(sys.argv[2])

# 作成したモデルを読み込む --- (*1)
FILE_MODEL = "bodydata_model.sav" 
cls = pickle.load(open(FILE_MODEL, 'rb'))

# 予測して結果を表示 --- (*2)
y_pred = cls.predict([[kg, cm]])
print("predict=", y_pred)

