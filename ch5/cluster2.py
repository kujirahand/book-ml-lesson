# -*- coding: utf-8 -*-
from sklearn import datasets
from sklearn import model_selection
from sklearn import cluster

# irisデータセット
iris = datasets.load_iris()
# 入力データ
in_data = iris.data

def mk_clusters(name, clf):
	# モデルを学習して適用
	predicted = clf.fit_predict(in_data)

	# 結果を表に
	result_clusters = [[0,0,0],[0,0,0],[0,0,0]]
	for i in range(len(predicted)):
		res_idx = predicted[i]
		tgt_idx = iris.target[i]
		result_clusters[res_idx][tgt_idx] += 1

	# 結果を表示
	print(name)
	print('\tiris0\tiris1\tiris2')
	print('class1\t'+'\t'.join(map(str,result_clusters[0])))
	print('class2\t'+'\t'.join(map(str,result_clusters[1])))
	print('class3\t'+'\t'.join(map(str,result_clusters[2]))+'\n')

# MiniBatchKMeans法
clf = cluster.MiniBatchKMeans(n_clusters=3)
mk_clusters('MiniBatchKMeans', clf)

# SpectralClustering法
clf = cluster.SpectralClustering(n_clusters=3)
mk_clusters('SpectralClustering', clf)

# Birch法
clf = cluster.Birch(n_clusters=3)
mk_clusters('Birch', clf)

# MeanShift法
clf = cluster.MeanShift()
mk_clusters('MeanShift', clf)

# DBSCAN法
clf = cluster.DBSCAN()
mk_clusters('DBSCAN', clf)
