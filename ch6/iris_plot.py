# -*- coding: utf-8 -*-
from sklearn import datasets as skdata
from sklearn import model_selection
import chainer
from chainer import training, datasets, iterators, optimizers
from chainer.training import extensions
import chainer.links as L
import numpy as np
from PIL import Image
from PIL import ImageDraw

import iris_network2 as neuralnet

# irisデータセット
iris = skdata.load_iris()

# ニューラルネットワークのモデルを読み込む
plot_net = neuralnet.Plot_NN()
chainer.serializers.load_npz( 'plot_model.npz', plot_net )

# 次元縮退を行う
in_data = iris.data
batch = np.array(in_data, dtype=np.float32)
result = plot_net.plot(batch).data

# 3個分の結果を表示
for i in range(3):
	print(str(iris.data[i]) + '\t =>\t ' + str(result[i]))

# 結果を画像にして保存
size = 150
colors = [(0xff,0,0),(0,0xff,0),(0,0,0xff)]
im = Image.new('RGB', (size,size), (0xff,0xff,0xff))
draw = ImageDraw.Draw(im)
# 結果内の最小値・最大値
mins = (min(result[:,0]), min(result[:,1]))
maxs = (max(result[:,0]), max(result[:,1]))
# 全ての結果を画像内にプロット
for i in range(len(in_data)):
	# 座標を求める
	x = int((result[i][0] - mins[0]) * size / (maxs[0] - mins[0]))
	y = int((result[i][1] - mins[1]) * size / (maxs[1] - mins[1]))
	# データの種類
	t = iris.target[i]
	# 文字を描写
	draw.text((x, y), str(t), colors[t])
# 画像を保存
im.save('result.png', 'PNG')

