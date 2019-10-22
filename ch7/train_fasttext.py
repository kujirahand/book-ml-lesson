import numpy as np
import fasttext

# 学習データを読み込む
clz1txt = open('1.txt').readlines()
clz2txt = open('3.txt').readlines()
clz3txt = open('5.txt').readlines()

# 全部繋げる
alltxt = []
alltxt.extend([(1,s) for s in clz1txt])
alltxt.extend([(2,s) for s in clz2txt])
alltxt.extend([(3,s) for s in clz3txt])

# __label__<クラス番号> で始まるファイルを作成する
with open('train.txt', 'w') as f:
	rnd_idx = np.random.permutation(len(alltxt))
	for i in rnd_idx:
		f.write('__label__%d\t%s' % alltxt[i])

# Fasttextを学習する
model = fasttext.train_supervised(
	input='train.txt', epoch=100, loss="hs")
model.save_model('fasttext.model')
