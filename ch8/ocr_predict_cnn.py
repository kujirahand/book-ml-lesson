import sys, math, cv2, pprint
import numpy as np
import chainer
import chainer.functions as F
import ocr_mlp as ocr

# ラベルを取得し、ラベル番号:文字コードに直す
all_labels = np.load('ocr-label.npy').tolist()
labels = {}
for k,v in all_labels.items():
	labels[v] = k
print("size=", len(all_labels))

# ニューラルネットワークのモデルを読み込む
ocr_net = ocr.OCR_NN(len(all_labels))
chainer.serializers.load_npz( 'ocr_model_cnn.npz', ocr_net )

# 引数からOCRシートの画像を取得
if len(sys.argv) <= 1:
	print("[USAGES] python ocr_predict_cnn.py imagefile")
	quit()
original = cv2.imread(sys.argv[1])
scale = 720.0 / original.shape[1]
resize_w = int(scale * original.shape[1])
resize_h = int(scale * original.shape[0])
src_img = cv2.resize(original, (resize_w, resize_h))
gray_img = cv2.cvtColor(src_img, cv2.COLOR_BGR2GRAY)

# マーカーを認識する
markers = []
for i in range(1,4):
	marker = cv2.imread('marker'+str(i)+'.png',0)
	matchs = cv2.matchTemplate(gray_img, marker, cv2.TM_CCOEFF_NORMED)
	min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(matchs)
	markers.append((max_loc[0],max_loc[1],marker.shape[1],marker.shape[0]))

# マーカー1と2の角度を求める
w_12 = markers[1][0] - markers[0][0]
h_12 = markers[1][1] - markers[0][1]
angle = math.atan2(h_12, w_12) * 180.0 / math.pi
# マーカー1と2、1と3の距離を求める
w_13 = markers[2][0] - markers[0][0]
h_13 = markers[2][1] - markers[0][1]
dst_width = int(math.sqrt(w_12*w_12 + h_12*h_12))
dst_height = int(math.sqrt(w_13*w_13 + h_13*h_13))
# OCRエリアのサイズを計算
x1 = markers[0][0]
y1 = markers[0][1]
x2 = x1 + dst_width + markers[1][2]
y2 = y1 + dst_height + markers[2][3]
# 画像の回転を直す
matrix = cv2.getRotationMatrix2D((x1,y1), angle, 1.0)
rotate_img = cv2.warpAffine(src_img, matrix,
	(src_img.shape[1],src_img.shape[0]))
# OCRエリアを切りだす
crop_img = rotate_img[y1:y2, x1:x2]
cv2.imwrite("hoge.png", crop_img)

# OCRエリア内の文字の位置（エリアサイズ比）
x_pos = [0.071197411003236,0.160194174757282,
		0.247572815533981,0.33495145631068,
		0.420711974110032,0.506472491909385,
		0.593851132686084,0.679611650485437,
		0.766990291262136,0.854368932038835]
y_pos = 0.423387096774194
w_pos = 0.067961165048544
h_pos = 0.169354838709677
# OCRシートの赤色を除く範囲を指定する
mask_color = (np.array([0, 0, 0]),np.array([255, 100, 100])) # 明度・彩度で指定
# 画像のリスト
predict_batch = []
# 文字を切りだす
for i in range(len(x_pos)):
	# 文字の場所を計算
	pos_x1 = int(x_pos[i] * (x2 - x1))
	pos_y1 = int(y_pos * (y2 - y1))
	pos_x2 = pos_x1 + int(w_pos * (x2 - x1))
	pos_y2 = pos_y1 + int(h_pos * (y2 - y1))
	# 文字の画像を切りだす
	char_img = crop_img[pos_y1:pos_y2, pos_x1:pos_x2]
	# 白黒画像にする
	char_gray = cv2.cvtColor(char_img, cv2.COLOR_BGR2GRAY)
	char_otsu = cv2.threshold(char_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
	char_bin = char_otsu[1]
	# カラーモデルを変換してマスクをかける
	hsv_img = cv2.cvtColor(char_img, cv2.COLOR_BGR2HSV)
	mask_img = cv2.inRange(hsv_img, mask_color[0], mask_color[1])
	char_img = cv2.bitwise_and(255 - char_bin, 255 - char_bin, mask=mask_img)
	# 画像認識用のサイズにする
	predict_img = cv2.resize(char_img, (64, 64))
	cv2.imwrite("ch" + str(i) + ".jpg", predict_img)
	predict_batch.append(predict_img.reshape(1, 64, 64))

# 画像認識を行う
predict_pixel = np.array(predict_batch, dtype=np.float32)
with chainer.using_config('train', False):
	batch = F.softmax(ocr_net(predict_pixel))
	# 画像認識の結果を表示する
	for i in range(len(batch.data)):
		index = np.argmax(batch[i].data)
		code = labels[index]
		ch = bytearray([code]).decode('sjis')
		print(ch, '(score:' + str(batch[i][index]) + ')')
