# -*- coding: utf-8 -*-
import chainer
import chainer.functions as F
import chainer.links as L

# ニューラルネットワークのモデル
class Iris_NN(chainer.Chain):

	def __init__(self):
		super(Iris_NN, self).__init__()
		with self.init_scope():
			self.c1 = L.Linear(4, 32)
			self.c2 = L.Linear(32, 32)
			self.c3 = L.Linear(32, 32)
			self.c4 = L.Linear(32, 32)
			self.c5 = L.Linear(32, 3)

	def __call__(self, x):
		h1 = F.leaky_relu(self.c1(x))
		h2 = F.leaky_relu(self.c2(h1))
		h3 = F.leaky_relu(self.c3(h2))
		h4 = F.leaky_relu(self.c4(h3))
		return self.c5(h4)
