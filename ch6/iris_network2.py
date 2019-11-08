# -*- coding: utf-8 -*-
import chainer
import chainer.functions as F
import chainer.links as L

# ニューラルネットワークのモデル
class Plot_NN(chainer.Chain):

	def __init__(self):
		super(Plot_NN, self).__init__()
		with self.init_scope():
			self.c1 = L.Linear(4, 32)
			self.c2 = L.Linear(32, 32)
			self.c3 = L.Linear(32, 32)
			self.c4 = L.Linear(32, 32)
			self.c5 = L.Linear(32, 2)
			self.c6 = L.Linear(2, 32)
			self.c7 = L.Linear(32, 32)
			self.c8 = L.Linear(32, 32)
			self.c9 = L.Linear(32, 32)
			self.c10 = L.Linear(32, 4)

	def __call__(self, x):
		h1 = F.leaky_relu(self.c1(x))
		h2 = F.leaky_relu(self.c2(h1))
		h3 = F.leaky_relu(self.c3(h2))
		h4 = F.leaky_relu(self.c4(h3))
		h5 = F.leaky_relu(self.c5(h4))
		h6 = F.leaky_relu(self.c6(h5))
		h7 = F.leaky_relu(self.c7(h6))
		h8 = F.leaky_relu(self.c8(h7))
		h9 = F.leaky_relu(self.c9(h8))
		return self.c10(h9)

	def plot(self, x):
		h1 = F.leaky_relu(self.c1(x))
		h2 = F.leaky_relu(self.c2(h1))
		h3 = F.leaky_relu(self.c3(h2))
		h4 = F.leaky_relu(self.c4(h3))
		return self.c5(h4)
