# -*- coding: utf-8 -*-
from sklearn import datasets
from sklearn import model_selection
from sklearn import neighbors

# irisデータセット
iris = datasets.load_iris()

# 訓練データとテストデータに分割
train_data, test_data, train_target, test_target \
	= model_selection.train_test_split(iris.data, iris.target, test_size=0.3, random_state=1)

# モデルを学習
clf = neighbors.KNeighborsClassifier(n_neighbors=6)
clf.fit(train_data, train_target)

# モデルの検証
score = clf.score(test_data, test_target)
print('score=' + str(score))

# 予測
predicted = clf.predict([[4.0, 2.0, 3.0, 1.0]])
print('predict=' + str(predicted))

