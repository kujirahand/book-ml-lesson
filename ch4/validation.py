# -*- coding: utf-8 -*-
from sklearn import datasets
from sklearn import model_selection
from sklearn import svm
from sklearn import neighbors
from sklearn import tree
from sklearn import ensemble

# irisデータセット
iris = datasets.load_iris()

# 交差検証の分割数
splits = 5
# 交差検証のスコア
score_linier = 0
score_poly = 0
score_rbf = 0
score_kneighbors = 0
score_dtree = 0
score_randomfr = 0

# 訓練データとテストデータに分割
for train_idx, test_idx in model_selection.KFold(n_splits=splits).split(iris.data):
	train_data = iris.data[train_idx]
	train_target = iris.target[train_idx]
	test_data = iris.data[test_idx]
	test_target = iris.target[test_idx]

	# モデルを学習
	clf1 = svm.LinearSVC(max_iter=10000)
	clf1.fit(train_data, train_target)
	clf2 = svm.SVC(kernel='poly', degree=3, gamma='scale')
	clf2.fit(train_data, train_target)
	clf3 = svm.SVC(kernel='rbf', gamma='scale')
	clf3.fit(train_data, train_target)
	clf4 = neighbors.KNeighborsClassifier(n_neighbors=6)
	clf4.fit(train_data, train_target)
	clf5 = tree.DecisionTreeClassifier(max_depth=3)
	clf5.fit(train_data, train_target)
	clf6 = ensemble.RandomForestClassifier(n_estimators=10)
	clf6.fit(train_data, train_target)

	# モデルの検証
	score_linier += clf1.score(test_data, test_target)
	score_poly += clf2.score(test_data, test_target)
	score_rbf += clf3.score(test_data, test_target)
	score_kneighbors += clf4.score(test_data, test_target)
	score_dtree += clf5.score(test_data, test_target)
	score_randomfr += clf6.score(test_data, test_target)

# 結果を表示
print('Linier\t= ' + str(score_linier / splits))
print('Poly\t= ' + str(score_poly / splits))
print('RBF\t= ' + str(score_rbf / splits))
print('K-Neighbors\t= ' + str(score_kneighbors / splits))
print('DecisionTree\t= ' + str(score_dtree / splits))
print('RandomForest\t= ' + str(score_randomfr / splits))

