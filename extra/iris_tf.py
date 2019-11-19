from sklearn import datasets
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.utils import np_utils

# scikit-learn からIrisのデータを読み込む
iris = datasets.load_iris()
X = iris.data
y = np_utils.to_categorical(iris.target)

#  データを学習用とテスト用に分割
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3)

# MLPモデルを構築
model = Sequential()
model.add(Dense(
    input_dim=4, output_dim=100, 
    bias=True, activation='relu'))
model.add(Dense(
    input_dim=100, output_dim=3, 
    bias=True, activation='softmax'))
model.compile(
    loss='categorical_crossentropy', 
    optimizer='sgd', metrics=['accuracy'])

# データの学習
model.fit(X_train, y_train,
    batch_size=32, epochs=100, verbose=1,
    validation_data=(X_test, y_test))

# 結果を表示
train_score = model.evaluate(X_train, y_train)
test_score = model.evaluate(X_test, y_test)
print('train loss:', train_score[0])
print('test loss:', test_score[0])
print('train accuracy:', train_score[1])
print('test accuracy:', test_score[1])

