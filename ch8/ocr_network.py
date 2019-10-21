import chainer
import chainer.functions as F
import chainer.links as L

# ニューラルネットワークのモデル
class OCR_NN(chainer.Chain):

	def __init__(self, num_labels):
		super(OCR_NN, self).__init__()
		with self.init_scope():
			self.c1 = L.Convolution2D(1, 32, ksize=3, stride=1)
			self.c2 = L.Convolution2D(32, 32, ksize=3, stride=1)
			self.c3 = L.Convolution2D(32, 64, ksize=3, stride=1)
			self.c4 = L.Convolution2D(64, 64, ksize=3, stride=1)
			self.c5 = L.Convolution2D(64, 64, ksize=3, stride=1)
			self.c6 = L.Convolution2D(64, 96, ksize=3, stride=1)
			self.c7 = L.Convolution2D(96, 96, ksize=3, stride=1)
			self.c8 = L.Linear(3*3*96, num_labels)

	def __call__(self, x):
		h1 = F.leaky_relu(self.c1(x))
		h2 = F.leaky_relu(self.c2(h1))
		h3 = F.max_pooling_2d(h2, ksize=3, stride=3)
		h4 = F.leaky_relu(self.c3(h3))
		h5 = F.leaky_relu(self.c4(h4))
		h6 = F.leaky_relu(self.c5(h5))
		h7 = F.max_pooling_2d(h6, ksize=2, stride=2)
		h8 = F.leaky_relu(self.c6(h7))
		h9 = F.leaky_relu(self.c7(h8))
		return self.c8(h9)
