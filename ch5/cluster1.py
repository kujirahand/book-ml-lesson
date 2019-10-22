# -*- coding: utf-8 -*-
from sklearn import datasets
from sklearn import model_selection
from sklearn import cluster

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

# 結果を表示
print('\tiris0\tiris1\tiris2')
print('class1\t'+'\t'.join(map(str,result_clusters[0])))
print('class2\t'+'\t'.join(map(str,result_clusters[1])))
print('class3\t'+'\t'.join(map(str,result_clusters[2])))
