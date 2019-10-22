import fasttext
from sklearn.metrics import classification_report

# 学習済みモデルを読み込む
classifier = fasttext.load_model('fasttext.model')

def read_text(fname):
    with open(fname, "r") as f:
        text = f.read()
        text = text.replace("\u3000", "")
        text = text.replace("\r\n", "\n")
        return text.split("\n")
        
# テスト用データを読み込む
clz1txt = read_text('2.txt')
clz2txt = read_text('4.txt')
clz3txt = read_text('6.txt')

# 結果として返されるラベル文字列
labels = {"__label__1":0, "__label__2":1, "__label__3":2}

# クラス分類を行う
label = classifier.predict(clz1txt)
clz1 = [labels[l[0]] for l in label[0]]
print("clz1:", clz1)

label = classifier.predict(clz2txt)
clz2 = [labels[l[0]] for l in label[0]]
print("clz2:", clz2)

label = classifier.predict(clz3txt)
clz3 = [labels[l[0]] for l in label[0]]
print("clz3:", clz3)

# クラス分類をの結果を表示する
all = clz1 + clz2 + clz3
clz = [0] * len(clz1) + [1] * len(clz2) + [2] * len(clz3)
report = classification_report(all, clz, target_names=['class1','class2','class3'])
print(report)
