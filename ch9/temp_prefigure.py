import os
import random
import chainer
from chainer import training, datasets, iterators, optimizers
from chainer.training import extensions
import chainer.links as L
import chainer.functions as F
import numpy as np

import temp_network as tpn

import csv

tempture_data = []

with open('data.csv', 'r', encoding='sjis') as file:
	reader = csv.reader(file)
	for i in range(5):
		header = next(reader)
	for row in reader:
		t = float(row[1])
		# -10〜40度の範囲を0〜1にする
		tempture_data.append((t + 10) / 50)

# 直近9ヶ月分を取得
tempture = tempture_data[len(tempture_data)-9:len(tempture_data)]
# RNN用のデータにする
temp_dataset = tpn.get_souceset(tempture, 9)

# ニューラルネットワークのモデルを作成
temp_net = tpn.Tempture_NN()
chainer.serializers.load_npz( 'tempture_model.npz', temp_net )
# 一番最近のデータを取得してバッチサイズ=1分の配列にする
in_data = np.array([temp_dataset[len(temp_dataset)-1][0]], dtype=np.float32)
# 9ヶ月分のデータを入力する
result = temp_net(in_data)
# 8月分の気温が返されるので、最後を取得
last_data = result[7]
# 次の15ヶ月分の予想を行う
for i in range(15):
	# 結果を取得して表示する
	data = last_data.data[0]
	temp = (data*50.0 - 10.0)
	print('+%dmon:\t%f'%(i, temp))
	# バッチサイズ=1,1ヶ月分の配列にする
	in_data = np.array([[[data]]], dtype=np.float32)
	# 次の日のデータを入力する
	result = temp_net(in_data, reset=False)
	last_data = result[0]



