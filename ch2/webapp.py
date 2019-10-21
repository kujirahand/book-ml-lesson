#!/usr/bin/env python3
import pickle
from sklearn import svm
from flask import Flask, request
app = Flask(__name__)

@app.route('/')
def root():
    # HTMLフォームを表示
    return html("""
    <div><form action="/check">
    体重(kg): <input name='kg'><br>
    身長(cm): <input name='cm'><br>
    <input type='submit' value='判定'>
    </form></div>
    """)

@app.route('/check')
def check():
    # パラメータを読む
    kg = request.args.get("kg")
    cm = request.args.get("cm")
    # 作成したモデルを読み込む --- (*1)
    FILE_MODEL = "bodydata_model.sav" 
    cls = pickle.load(open(FILE_MODEL, 'rb'))
    # 予測して結果を表示 --- (*2)
    y_pred = cls.predict([[kg, cm]])
    return html("""
    <h1>{}kg {}cm→{}</h1>
    <a href="/">戻る</a>
    """.format(kg, cm, y_pred[0]))

def html(body):
    return """
    <html><head><style>
    *   { padding:8px; margin:4px;  }
    div { border: 1px solid silver; }
    h1  { border-bottom: 3px solid silver; }
    </style></head><body>
    <h1>肥満判定</h1>
    """ + body + """
    </body></html>"""

# 起動
if __name__ == "__main__":
    app.run(debug=True, port=8080)


