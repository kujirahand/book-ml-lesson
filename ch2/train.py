from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# CSVファイルを読み込む --- (*1)
X = []
Y = []
csv = ""
with open("bodydata.csv", "rt") as f:
    csv = f.read()
for line in csv.split("\n"):
    cells = line.split(",")
    if len(cells) < 3: continue
    kg = float(cells[0])
    cm = float(cells[1])
    result = cells[2]
    X.append([kg, cm])
    Y.append(result)

# データを訓練用とテスト用に分割 --- (*2)
x_train, x_test, y_train, y_test = \
        train_test_split(X, Y)

# データを学習する --- (*3)
cls = svm.SVC(gamma=0.001)
cls.fit(x_train, y_train)

# 学習内容をテストデータで評価する --- (*4)
y_pred = cls.predict(x_test)
acc = accuracy_score(y_test, y_pred)
print("accuracy=", acc)

# 作成したモデルを保存 --- (*5)
import pickle
pickle.dump(cls, open("bodydata_model.sav", "wb"))
print("ok")

