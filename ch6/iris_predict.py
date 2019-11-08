# -*- coding: utf-8 -*-
import numpy as np
import chainer
import chainer.functions as F

import iris_network1 as neuralnet

# ニューラルネットワークのモデルを読み込む
iris_net = neuralnet.Iris_NN()
chainer.serializers.load_npz( 'iris_model.npz', iris_net )

# クラス分類を行う
predict_data = np.array([[4.0, 2.0, 3.0, 1.0]], dtype=np.float32)
batch = F.softmax(iris_net(predict_data))
# 画像認識の結果を表示する
for index in range(3):
	print(str(index) + '\tscore:' + str(batch[0][index]))
