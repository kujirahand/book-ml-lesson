# -*- coding: utf-8 -*-
import numpy as np
from sklearn import ensemble
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report

# 学習データを読み込む
clz1txt = open('1.txt').readlines()
clz2txt = open('3.txt').readlines()
clz3txt = open('5.txt').readlines()

# 全部繋げる
alltxt = []
alltxt.extend([s for s in clz1txt])
alltxt.extend([s for s in clz2txt])
alltxt.extend([s for s in clz3txt])

# クラスを作成
clazz = [0] * len(clz1txt) + [1] * len(clz2txt) + [2] * len(clz3txt)

# TF-IDFベクトル化
vectorizer = TfidfVectorizer(use_idf=True, token_pattern='(?u)\\b\\w+\\b')
vecs = vectorizer.fit_transform(alltxt)

# モデルを学習
clf = ensemble.RandomForestClassifier(n_estimators=10)
clf.fit(vecs, clazz)

# テスト用データを読み込む
tst1txt = open('2.txt').readlines()
tst2txt = open('4.txt').readlines()
tst3txt = open('6.txt').readlines()

# TF-IDFベクトル化
vec1 = vectorizer.transform(tst1txt)
vec2 = vectorizer.transform(tst2txt)
vec3 = vectorizer.transform(tst3txt)

# クラス分類を行う
clz1 = clf.predict(vec1)
clz1 = list(clz1)
print('clz1='+str(clz1))

clz2 = clf.predict(vec2)
clz2 = list(clz2)
print('clz2='+str(clz2))

clz3 = clf.predict(vec3)
clz3 = list(clz3)
print('clz3='+str(clz3))

# クラス分類をの結果を表示する
all = clz1 + clz2 + clz3
clz = [0] * len(clz1) + [1] * len(clz2) + [2] * len(clz3)
report = classification_report(all, clz, target_names=['class1','class2','class3'])
print(report)
