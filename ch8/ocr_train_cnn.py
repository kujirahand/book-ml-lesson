import pprint
from PIL import Image, ImageOps
import os, random, chainer
from chainer import training, datasets, iterators, optimizers
from chainer.training import extensions
import chainer.links as L

# 画像の回転を使うかどうか
USE_ROTATE_IMAGE = False

# GPUを使用する場合のみ
USE_GPU = False
if USE_GPU:
    import cupy as xp
    import chainer.cuda
    # GPU利用時のメモリエラーを防ぐ
    pool = xp.cuda.MemoryPool(xp.cuda.malloc_managed)
    xp.cuda.set_allocator(pool.malloc)
    # GPU#0を使う
    chainer.cuda.get_device_from_id(0).use()
else:
    # GPUを使用しない場合
    import numpy as xp

# ETL1のデータを読むためのライブラリ
import etl1
# ネットワーク構造の定義
import ocr_network as ocr

# 画像データとラベル番号を保持する配列
ocr_dataset = []
test_ocr_dataset = []
# 全てのラベル
all_labels = {}

def image_to_data(image, code):
    # 大きさを揃える
    img64 = image.resize((64, 64))
    # 画像を数値配列に
    pixels = xp.array(img64, dtype=xp.float32)
    pixels = pixels.reshape((1, 64, 64))
    pixels /= 255
    # インデックスを決める
    if code in all_labels:
        label = all_labels[code]
    else:
        label = len(all_labels)
        all_labels[code] = label
    return (pixels, xp.int32(label))

# 画像を読み取る
for i, cur in enumerate(etl1.get_image()):
    image, code = cur
    if USE_ROTATE_IMAGE:
        # 角度を変えて登録する
        for rad in range(-2, 3, 2):
            im_r = image.rotate(rad)
            data = image_to_data(im_r, code)
            if i % 10 != 0:
                ocr_dataset.append(data)
            else:
                test_ocr_dataset.append(data)
    else:
        data = image_to_data(image, code)
        if i % 10 != 0:
            ocr_dataset.append(data)
        else:
            test_ocr_dataset.append(data)

# ラベルをファイルに保存
xp.save('ocr-label.npy', all_labels)
print("label size=" , len(all_labels))
print("train dataset size=", len(ocr_dataset))
print("test  dataset size=", len(test_ocr_dataset))

# 配列をChainerのイテレーターにする
batch_size = 64
train_iter = iterators.SerialIterator(ocr_dataset, batch_size)
test_iter = iterators.SerialIterator(test_ocr_dataset, batch_size, repeat=False)
# ニューラルネットワークのモデルを作成
ocr_net = ocr.OCR_NN(len(all_labels))
model = L.Classifier(ocr_net)
if USE_GPU:
    model.to_gpu()  # GPUを使用する場合のみ
# 学習アルゴリズムの選択
optimizer = chainer.optimizers.RMSpropGraves()
optimizer.setup(model)
# GPU#0を使用して学習モデルを作成
if USE_GPU:
    updater = training.StandardUpdater(train_iter, optimizer, device=0)
else:
    updater = training.StandardUpdater(train_iter, optimizer)
# 25エポック分学習させる
trainer = training.Trainer(updater, (25, 'epoch'), out="result")
# GPU#0を使用してテストを実行
if USE_GPU:
    trainer.extend(extensions.Evaluator(test_iter, model, device=0))
else:
    trainer.extend(extensions.Evaluator(test_iter, model))

# 学習の進展を表示するようにする
trainer.extend(extensions.LogReport())
trainer.extend(extensions.ProgressBar(update_interval=10))
trainer.extend(extensions.PrintReport(['epoch','main/loss',
	'validation/main/loss','main/accuracy', 'validation/main/accuracy']))
# 機械学習を実行する
trainer.run()
# 学習結果を保存する
chainer.serializers.save_npz( 'ocr_model_cnn.npz', ocr_net )
