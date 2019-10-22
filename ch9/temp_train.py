import os
import chainer
from chainer import training, datasets, iterators, optimizers
from chainer.training import extensions
import chainer.links as L
import chainer.functions as F
import numpy as xp
import csv
import temp_network as tpn

tempture_data = []

with open('data.csv', 'r', encoding="sjis") as file:
    reader = csv.reader(file)
    for i in range(5):
        header = next(reader)
    for row in reader:
        t = float(row[1])
        # -10から40度の範囲を0から1にする
        tempture_data.append((t + 10) / 50)

# 直近48ヶ月分を除く
tempture_train = tempture_data[0:len(tempture_data)-48]
# 直近48ヶ月分をテスト用にする
tempture_test = tempture_data[len(tempture_data)-48:len(tempture_data)]
# 24ヶ月のウィンドウでRNN用のデータにする
temp_dataset = tpn.get_souceset(tempture_train, 24)
test_temp_dataset = tpn.get_souceset(tempture_test, 24)

# 損失関数を定義する
def loss_func(result, label):
    loss = 0
    for i in range(len(result)):
        # 次の月の気温との比較を求める
        loss += F.mean_squared_error(result[i], label[:,i])
    return loss

# 確度関数を定義する
def acc_func(result, label):
    acc = 0
    for i in range(len(result)):
        # 差の絶対値の平均
        dif = result[i].data - label[:,i]
        acc += 1.0 - xp.mean(xp.abs(dif))
    return acc / len(result)

# 配列をChainerのイテレーターにする
batch_size = 64
train_iter = iterators.SerialIterator(temp_dataset, batch_size)
test_iter = iterators.SerialIterator(test_temp_dataset, batch_size, repeat=False)
# ニューラルネットワークのモデルを作成
stock_net = tpn.Tempture_NN()
model = L.Classifier(stock_net, lossfun=loss_func, accfun=acc_func)
# 学習アルゴリズムの選択
optimizer = chainer.optimizers.RMSpropGraves()
optimizer.setup(model)
# 学習モデルを作成
updater = training.StandardUpdater(train_iter, optimizer)
# 25エポック分学習させる
trainer = training.Trainer(updater, (50, 'epoch'), out="result")
# テストを実行
trainer.extend(extensions.Evaluator(test_iter, model))
# 学習の進展を表示するようにする
trainer.extend(extensions.LogReport())
trainer.extend(extensions.ProgressBar(update_interval=10))
trainer.extend(extensions.PrintReport(['epoch','main/loss',
	'validation/main/loss','main/accuracy', 'validation/main/accuracy']))
# 機械学習を実行する
trainer.run()
# 学習結果を保存する
chainer.serializers.save_npz( 'tempture_model.npz', stock_net )

