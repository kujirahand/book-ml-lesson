# -*- coding: utf-8 -*-
from sklearn import datasets
from sklearn import model_selection
from sklearn import svm

# irisデータセット
iris = datasets.load_iris()

# 訓練データとテストデータに分割
train_data, test_data, train_target, test_target \
	= model_selection.train_test_split(iris.data, 
		iris.target, test_size=0.3, random_state=1)

# モデルを学習
clf1 = svm.SVC(kernel='poly', degree=3, gamma='scale')
clf1.fit(train_data, train_target)
# モデルを学習
clf2 = svm.SVC(kernel='rbf', gamma='scale')
clf2.fit(train_data, train_target)

# モデルの検証
score1 = clf1.score(test_data, test_target)
print('score1=' + str(score1))

# 予測
predicted1 = clf1.predict([[4.0, 2.0, 3.0, 1.0]])
print('predict1=' + str(predicted1))

# モデルの検証
score2 = clf2.score(test_data, test_target)
print('score2=' + str(score2))

# 予測
predicted2 = clf2.predict([[4.0, 2.0, 3.0, 1.0]])
print('predict2=' + str(predicted2))
