# -*- coding: utf-8 -*-
from sklearn import datasets
from sklearn import model_selection
from sklearn import cluster
from sklearn import metrics
import numpy as np

# irisデータセット
iris = datasets.load_iris()
# 入力データ
in_data = iris.data

# モデルを学習
clf = cluster.KMeans(n_clusters=3)
clf.fit(in_data)

# 予測
predicted = clf.predict(iris.data)

# 結果を表に
result_clusters = [[0,0,0],[0,0,0],[0,0,0]]
for i in range(len(predicted)):
	res_idx = predicted[i]
	tgt_idx = iris.target[i]
	result_clusters[res_idx][tgt_idx] += 1

# 結果のクラスと、アヤメの種類とのマッチングを行う
clazz_idx = [np.argmax(result_clusters[0]),
	np.argmax(result_clusters[1]),
	np.argmax(result_clusters[2])]

# 結果のクラスと、アヤメの種類の一致不一致を表にする
score_data = [[],[],[]]
score_target = [[],[],[]]
for i in range(len(predicted)):
	res_idx = predicted[i]
	tgt_idx = iris.target[i]
	# 一致度のマトリクスを作成する
	for j in range(3):
		if j == res_idx:
			score_data[j].append(1)
		else:
			score_data[j].append(0)
		if clazz_idx[j] == tgt_idx:
			score_target[j].append(1)
		else:
			score_target[j].append(0)
	result_clusters[res_idx][tgt_idx] += 1

# 精度を取得する
print('Precision score:')
pr_score1 = metrics.precision_score(score_data[0], score_target[0])
print('score1\t= '+str(pr_score1))
pr_score2 = metrics.precision_score(score_data[1], score_target[1])
print('score2\t= '+str(pr_score2))
pr_score3 = metrics.precision_score(score_data[2], score_target[2])
print('score3\t= '+str(pr_score3))
print('average\t= '+str((pr_score1 + pr_score2 + pr_score3) / 3.0))

# 再現率を取得する
print('Recall score:')
rc_score1 = metrics.recall_score(score_data[0], score_target[0])
print('score1\t= '+str(rc_score1))
rc_score2 = metrics.recall_score(score_data[1], score_target[1])
print('score2\t= '+str(rc_score2))
rc_score3 = metrics.recall_score(score_data[2], score_target[2])
print('score3\t= '+str(rc_score3))
print('average\t= '+str((rc_score1 + rc_score2 + rc_score3) / 3.0))

# F1値を取得する
print('F1 score:')
f1_score1 = metrics.f1_score(score_data[0], score_target[0])
print('score1\t= '+str(f1_score1))
f1_score2 = metrics.f1_score(score_data[1], score_target[1])
print('score2\t= '+str(f1_score2))
f1_score3 = metrics.f1_score(score_data[2], score_target[2])
print('score3\t= '+str(f1_score3))
print('average\t= '+str((f1_score1 + f1_score2 + f1_score3) / 3.0))
