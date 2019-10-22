from sklearn import datasets as skdata
from sklearn import model_selection
import chainer
from chainer import training, datasets, iterators, optimizers
from chainer.training import extensions
import chainer.links as L
import numpy as xp

# 学習データを読み込む
txt1 = open('1.txt').readlines()
txt2 = open('3.txt').readlines()
txt3 = open('5.txt').readlines()

# 単語IDのリストのデータ
train_data = []
train_target = []
# 語彙と単語IDのディクショナリ
voc = {}
n_voc = 0

# 単語IDのリストにデータを追加する関数
def add_txt(txt, c):
    global voc
    global n_voc
    for l in txt:
        v = []
        for w in l.split():
            if w in voc:
                z = voc[w]
            else:
                voc[w] = n_voc
                z = n_voc
                n_voc += 1
            v.append(z)
        train_data.append(v)
        train_target.append(c)

# 単語IDのリストにデータを追加する
add_txt(txt1, 0)
add_txt(txt2, 1)
add_txt(txt3, 2)

# Countベクトルとクラスのタプルにする
train_dataset = []
for i in range(len(train_data)):
    v = [0] * n_voc
    for p in train_data[i]:
        v[p] += 1
    train_dataset.append((
        xp.array(v, dtype=xp.float32), 
        xp.array(train_target[i], dtype=xp.int32)))


import text_network as neuralnet

# 配列をChainerのイテレーターにする
batch_size = 64
train_iter = iterators.SerialIterator(train_dataset, batch_size)
# ニューラルネットワークのモデルを作成
text_net = neuralnet.Text_NN(n_voc)
model = L.Classifier(text_net)
# 学習アルゴリズムの選択
optimizer = chainer.optimizers.RMSpropGraves()
optimizer.setup(model)
# 学習モデルを作成
updater = training.StandardUpdater(train_iter, optimizer)
# 35エポック分学習させる
trainer = training.Trainer(updater, (35, 'epoch'), out="result")
# 学習の進展を表示するようにする
trainer.extend(extensions.LogReport())
trainer.extend(extensions.ProgressBar(update_interval=10))
trainer.extend(extensions.PrintReport(['epoch','main/loss', 'main/accuracy']))
# 機械学習を実行する
trainer.run()
# 学習結果を保存する
chainer.serializers.save_npz( 'text_model.npz', text_net )
# 語彙と単語IDのディクショナリも保存する
import numpy as np
np.save('text_voc.npy', voc)

