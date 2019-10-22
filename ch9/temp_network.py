# -*- coding: utf-8 -*-
import chainer
import chainer.functions as F
import chainer.links as L
import numpy as np

# ニューラルネットワークのモデル
class Tempture_NN(chainer.Chain):

	def __init__(self):
		super(Tempture_NN, self).__init__()
		with self.init_scope():
			self.c1 = L.StatefulGRU(1, 32)
			self.c2 = L.StatefulGRU(32, 32)
			self.c3 = L.StatefulGRU(32, 10)

	def __call__(self, x, reset=True):
		# LSTMのステータスをクリアする
		if reset:
			self.c1.reset_state()
			self.c2.reset_state()
			self.c3.reset_state()
		# 入力された分を計算する
		result = []
		aa = ''
		for i in range(x.shape[1]):
			batch = x[:,i]
			h1 = self.c1(batch)
			h2 = self.c2(h1)
			h3 = self.c3(h2)
			# 結果の足し合わせをニューロンの数で平均する
			h4 = F.sum(h3, axis=1) / 10.0
			result.append(h4)
		return result

# 気温データからRNNの教師データセットを作成する関数
def get_souceset(tempture, size):
	result_data = []
	# 過去の全ての月に対して
	for i in range(len(tempture)-size+1):
		source = []
		result = []
		# 前月と来月を取得する
		for j in range(i, i+size-1):
			dif = np.float32(tempture[j])
			res = np.float32(tempture[j+1])
			add = np.array([dif], dtype=np.float32)
			source.append(add)
			result.append(res)
		result_data.append((source,result))
	return result_data

