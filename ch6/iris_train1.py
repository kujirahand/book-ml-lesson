from sklearn import datasets as skdata
from sklearn import model_selection
import chainer
from chainer import training, datasets, iterators, optimizers
from chainer.training import extensions
import chainer.links as L
import numpy as xp

import iris_network1 as neuralnet

# irisデータセット
iris = skdata.load_iris()

# 訓練データとテストデータに分割
train_data, test_data, train_target, test_target \
	= model_selection.train_test_split(
		iris.data, iris.target, 
		test_size=0.3, random_state=1)

# データと結果のタプルの配列にする
train_dataset = [(xp.float32(train_data[i]), 
	xp.int32(train_target[i]))
 	for i in range(len(train_data))]
test_dataset = [(xp.float32(test_data[i]), 
	xp.int32(test_target[i]))
 	for i in range(len(test_data))]

# 配列をChainerのイテレーターにする
batch_size = 64
train_iter = iterators.SerialIterator(
	train_dataset, batch_size)
test_iter = iterators.SerialIterator(
	test_dataset, batch_size, repeat=False)
# ニューラルネットワークのモデルを作成
iris_net = neuralnet.Iris_NN()
model = L.Classifier(iris_net)
# 学習アルゴリズムの選択
optimizer = chainer.optimizers.RMSpropGraves()
optimizer.setup(model)
# 学習モデルを作成
updater = training.StandardUpdater(train_iter, optimizer)
# 50エポック分学習させる
trainer = training.Trainer(updater, (50, 'epoch'), out="result")
# テストを実行
trainer.extend(extensions.Evaluator(test_iter, model))
# 学習の進展を表示するようにする
trainer.extend(extensions.LogReport())
trainer.extend(extensions.ProgressBar(update_interval=10))
trainer.extend(extensions.PrintReport(['epoch','main/loss',
	'validation/main/loss','main/accuracy', 
	'validation/main/accuracy']))
# 機械学習を実行する
trainer.run()
# 学習結果を保存する
chainer.serializers.save_npz( 'iris_model.npz', iris_net )
