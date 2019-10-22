# -*- coding: utf-8 -*-
from sklearn import datasets
from sklearn import model_selection
from sklearn import neural_network

# irisデータセット
iris = datasets.load_iris()
# 最初の三つのデータを説明変数に
data = iris.data[:,0:3]
# 最後の一つのデータを目的変数に
target = iris.data[:,3:4]

# 訓練データとテストデータに分割
train_data, test_data, train_target, test_target \
	= model_selection.train_test_split(
                data, target, 
                test_size=0.3, random_state=1)

# 配列の次元数を変更
train_target = train_target[:,0]
test_target = test_target[:,0]

# モデルを学習
clf = neural_network.MLPRegressor(
        hidden_layer_sizes=(5, 5), max_iter=10000)
clf.fit(train_data, train_target)

# モデルの検証
score = clf.score(test_data, test_target)
print('score=' + str(score))

# 予測
predicted = clf.predict([[4.0, 2.0, 3.0]])
print('predict=' + str(predicted))

