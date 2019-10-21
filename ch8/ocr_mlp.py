import chainer
import chainer.links as L
import chainer.functions as F

class OCR_NN(chainer.Chain):

    def __init__(self, n_out=10, n_mid_units=100):
        super(OCR_NN, self).__init__()
        # パラメータを持つ層の登録
        with self.init_scope():
            self.l1 = L.Linear(None, n_mid_units)
            self.l2 = L.Linear(n_mid_units, n_mid_units)
            self.l3 = L.Linear(n_mid_units, n_out)

    def __call__(self, x):
        # データを受け取った際のforward計算を書く
        h1 = F.relu(self.l1(x))
        h2 = F.relu(self.l2(h1))
        return self.l3(h2)
