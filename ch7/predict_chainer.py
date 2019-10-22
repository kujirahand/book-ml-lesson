# -*- coding: utf-8 -*-
import numpy as np
import chainer
import chainer.functions as F
from sklearn.metrics import classification_report
import text_network as neuralnet

# テストデータを読み込む
txt1 = open('2.txt').readlines()
txt2 = open('4.txt').readlines()
txt3 = open('6.txt').readlines()

# 語彙と単語IDのディクショナリを読み込む
voc = np.load('text_voc.npy').item()
n_voc = max(voc.values()) + 1

# 単語IDのリストのデータ
predict_data = []

# 単語IDのリストにデータを追加する関数
def add_txt(txt):
	for l in txt:
		v = []
		for w in l.split():
			if w in voc:
				z = voc[w]
				v.append(z)
		predict_data.append(v)

# 単語IDのリストにデータを追加する
add_txt(txt1)
add_txt(txt2)
add_txt(txt3)

# Countベクトルにする
predict_dataset = []
for i in range(len(predict_data)):
	v = [0] * n_voc
	for p in predict_data[i]:
		v[p] += 1
	predict_dataset.append(v)

# ニューラルネットワークのモデルを作成
text_net = neuralnet.Text_NN(n_voc)
chainer.serializers.load_npz( 'text_model.npz', text_net )
# クラス分類を行う
predict_dataset = np.array(predict_dataset, dtype=np.float32)
batch = F.softmax(text_net(predict_dataset))
# クラス分類をの結果を表示する
all = [np.argmax(batch[i].data) for i in range(len(batch))]
clz = [0] * len(txt1) + [1] * len(txt2) + [2] * len(txt3)
report = classification_report(all, clz, target_names=['class1','class2','class3'])
print(report)


