# -*- coding: utf-8 -*-
from sklearn import datasets
from sklearn import model_selection
from sklearn import decomposition
from PIL import Image
from PIL import ImageDraw


# irisデータセット
iris = datasets.load_iris()
# 入力データ
in_data = iris.data

# 次元数を2に削減
dec = decomposition.TruncatedSVD(n_components=2)
result = dec.fit_transform(in_data)

# 3個分の結果を表示
for i in range(3):
	print(str(in_data[i]) + '\t =>\t ' + str(result[i]))

# 結果を画像にして保存
size = 150
colors = [(0xff,0,0),(0,0xff,0),(0,0,0xff)]
im = Image.new('RGB', (size,size), (0xff,0xff,0xff))
draw = ImageDraw.Draw(im)
# 結果内の最小値・最大値
mins = (min(result[:,0]), min(result[:,1]))
maxs = (max(result[:,0]), max(result[:,1]))
# 全ての結果を画像内にプロット
for i in range(len(in_data)):
	# 座標を求める
	x = int((result[i][0] - mins[0]) * size / (maxs[0] - mins[0]))
	y = int((result[i][1] - mins[1]) * size / (maxs[1] - mins[1]))
	# データの種類
	t = iris.target[i]
	# 文字を描写
	draw.text((x, y), str(t), colors[t])
# 画像を保存
im.save('result1.png', 'PNG')
