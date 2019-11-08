from sklearn import datasets as skdata
from sklearn import model_selection
import chainer
from chainer import training, datasets, iterators, optimizers
from chainer.training import extensions
import chainer.links as L
import chainer.functions as F
import numpy as xp

import iris_network2 as neuralnet

# irisデータセット
iris = skdata.load_iris()

# データとデータのタプルの配列にする
train_dataset = [(xp.float32(iris.data[i]), 
	xp.float32(iris.data[i])) 
	for i in range(len(iris.data))]

# 配列をChainerのイテレーターにする
batch_size = 64
train_iter = iterators.SerialIterator(train_dataset, batch_size)
# ニューラルネットワークのモデルを作成
plot_net = neuralnet.Plot_NN()
model = L.Classifier(plot_net, lossfun=F.mean_absolute_error, accfun=F.mean_absolute_error)
# 学習アルゴリズムの選択
optimizer = chainer.optimizers.RMSpropGraves()
optimizer.setup(model)
# GPU#0を使用して学習モデルを作成
updater = training.StandardUpdater(train_iter, optimizer)
# 500エポック分学習させる
trainer = training.Trainer(updater, (500, 'epoch'), out="result")
# 学習の進展を表示するようにする
trainer.extend(extensions.LogReport())
trainer.extend(extensions.ProgressBar(update_interval=10))
trainer.extend(extensions.PrintReport(['epoch','main/loss']))
# 機械学習を実行する
trainer.run()
# 学習結果を保存する
chainer.serializers.save_npz( 'plot_model.npz', plot_net )
